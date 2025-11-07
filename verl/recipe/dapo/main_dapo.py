# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
注意，我们没有将 main 和 ray_trainer 合并，因为 ray_trainer 被其他 main 使用。
"""

import hydra
import ray

from verl.trainer.ppo.reward import get_custom_reward_fn

from .dapo_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    # 调用 run_ppo 函数
    run_ppo(config)


def run_ppo(config) -> None:
    # 检查 Ray 是否已经初始化，如果未初始化则进行初始化
    if not ray.is_initialized():
        # 这是用于本地 Ray 集群的配置
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # 创建任务执行器
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # 请确保 main_task 不会被调度到主节点上
class TaskRunner:
    def run(self, config):
        # 打印初始配置
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # 打印并解析配置文件中的值
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True 会解析符号值
        OmegaConf.resolve(config)

        # 从 hdfs 下载检查点
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # 实例化 tokenizer（分词器）
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # 用于多模态 LLM，可能为 None

        # 定义工作者类
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            # 确保 actor 和 critic 使用相同的策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # 确保 actor 和 critic 使用相同的策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 映射角色与工作者
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # 定义资源池
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 如果启用奖励模型，则使用多源奖励函数
        # - 对于基于规则的奖励模型，我们直接调用奖励分数
        # - 对于基于模型的奖励模型，我们调用一个模型
        # - 对于代码相关的提示，如果有测试用例，我们发送到沙盒
        # - 最后，我们将所有奖励组合在一起
        # - 奖励类型取决于数据的标签
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 如果使用 KL 损失或策略参考模型，则设置参考模型
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 选择奖励管理器
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError

        # 获取自定义奖励函数
        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # 对于验证，始终使用基于函数的 RM
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        # 创建资源池管理器
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 创建训练器并开始训练
        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    # 运行 main 函数
    main()
