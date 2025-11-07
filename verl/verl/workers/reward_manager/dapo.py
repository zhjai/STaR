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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class DAPORewardManager:
    """奖励管理器类，用于计算奖励分数。"""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        """
        初始化奖励管理器。

        参数：
        - tokenizer: 分词器，用于解码响应。
        - num_examine: 打印多少批次的解码响应。
        - compute_score: 可选的自定义计算奖励分数的函数，默认为 default_compute_score。
        - reward_fn_key: 用于从数据中提取奖励函数的键，默认为 "data_source"。
        - max_resp_len: 最大响应长度，用于限制响应的长度。
        - overlong_buffer_cfg: 配置过长缓冲区的处理（如果有）。
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # 打印多少批次的解码响应
        self.compute_score = compute_score or default_compute_score  # 使用默认的奖励计算函数（如果没有提供）
        self.reward_fn_key = reward_fn_key  # 奖励函数键
        self.overlong_buffer_cfg = overlong_buffer_cfg  # 过长缓冲区配置
        self.max_resp_len = max_resp_len  # 最大响应长度

        # 如果启用了过长缓冲区配置，确保提供了 max_resp_len
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len 必须提供，如果启用了 {overlong_buffer_cfg=}, 但得到 None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        计算奖励分数。如果数据中有 rm_scores，则直接返回。
        否则，使用 compute_score 函数来计算奖励。

        参数：
        - data: 包含数据的 DataProto 实例。
        - return_dict: 如果为 True，返回包含奖励分数和额外信息的字典；否则，返回奖励张量。

        返回：
        - 奖励张量或包含奖励和额外信息的字典。
        """
        # 如果数据中已有奖励分数，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # 创建一个与响应形状相同的零张量用于存储奖励分数
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)  # 用于存储额外的奖励信息

        already_print_data_sources = {}  # 记录已打印的 data source

        # 遍历数据中的每个项
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # 获取提示词和响应
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()  # 有效提示词的长度
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()  # 有效响应的长度
            valid_response_ids = response_ids[:valid_response_length]

            # 解码提示词和响应
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token  # EOS（结束）标记
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]  # 去除 EOS 标记

            # 获取数据源、ground truth 和额外信息
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # 使用 compute_score 计算奖励分数
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # 如果结果是字典，提取分数并存储额外信息
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # 存储额外的奖励信息
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            # 计算最终奖励
            reward = score

            # 处理过长缓冲区配置（如果有）
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)  # 过长惩罚
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # 将奖励分数存储在张量中
            reward_tensor[i, valid_response_length - 1] = reward

            # 打印提示词、响应和奖励信息
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # 根据 return_dict 的值返回奖励张量或字典
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
