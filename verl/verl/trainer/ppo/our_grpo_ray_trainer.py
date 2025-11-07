# GRPO Ray Trainer Implementation  # Module title/file description

import uuid  # Generate unique IDs for trajectories/samples
from collections import defaultdict  # Dict with default factory for aggregation
from copy import deepcopy  # Deep copy utility (for complex structures)
from pprint import pprint  # Pretty print (e.g., validation metrics)

import numpy as np  # Numerical array operations
import torch  # PyTorch for tensor and DL ops
from tqdm import tqdm  # Progress bar

from verl import DataProto  # VERL data container for tensor and non-tensor batches
from verl.trainer.ppo.core_algos import agg_loss  # Aggregate loss by mask/strategy
from verl.trainer.ppo.metric_utils import (  # Metric utilities
    compute_data_metrics,  # Data-related metrics from batch
    compute_throughout_metrics,  # Throughput metrics
    compute_timing_metrics,  # Timing metrics
    reduce_metrics,  # Reduce metrics across GPUs/workers
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask  # Base Ray PPO trainer and helpers


class RayGRPOTrainer(RayPPOTrainer):  # Define GRPO trainer inheriting RayPPOTrainer
    """
    Ray-based GRPO Trainer using a single controller.
    Note: This trainer runs in the driver process on a single CPU/GPU node.
    """  # Class docstring; runtime topology

    def __init__(  # Constructor; initialize trainer
        self,
        config,  # Training config (OmegaConf/dict): algo/opt/logging
        tokenizer,  # Tokenizer for text->ids
        role_worker_mapping: dict,  # Role to remote worker mapping (actor/critic/rm)
        resource_pool_manager,  # Resource manager (e.g., number of GPUs)
        ray_worker_group_cls=None,  # Worker group class (customizable)
        processor=None,  # Preprocessor
        reward_fn=None,  # Token-level reward function (base)
        global_reward_fn=None,  # Global (sequence-level) reward function (GRPO)
        val_reward_fn=None,  # Reward function for validation
        train_dataset=None,  # Training dataset
        val_dataset=None,  # Validation dataset
        collate_fn=None,  # Collate function
        train_sampler=None,  # Sampler (e.g., distributed)
        device_name="cuda",  # Device name (default CUDA)
    ):
        super().__init__(  # Call parent RayPPOTrainer to init common components
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        self.global_reward_fn = global_reward_fn  # Store global reward function for training

        # ---- filter_groups persistent state (threshold-1: skip prompts permanently once hit) ----
        self._fg_persist_block = set()  # Prompt keys hit by threshold-1 (unique identifiers)
        self._fg_x1_total_unique = 0    # Total unique prompts dropped by threshold-1 (deduplicated)

    def fit(self):  # Main training loop
        """GRPO training loop"""  # Method doc
        from omegaconf import OmegaConf  # Lazy import to avoid top-level deps
        from verl.utils.tracking import Tracking  # Tracking logger

        # Initialize logger for tracking
        logger = Tracking(  # Create tracker to log metrics (e.g., W&B/MLflow)
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),  # Materialize config for logging
        )

        self.global_steps = 0  # Global step counter

        # Load checkpoint if available
        self._load_checkpoint()  # Restore model/optimizer/steps if checkpoint exists

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):  # Validate first if required
            val_metrics = self._validate()  # Run validation and get metrics
            assert val_metrics, f"{val_metrics=}"  # Sanity check
            pprint(f"Initial validation metrics: {val_metrics}")  # Print initial validation metrics
            logger.log(data=val_metrics, step=self.global_steps)  # Log to backend
            if self.config.trainer.get("val_only", False):  # Validation-only mode
                return  # Exit early

        # Training progress bar
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="GRPO Training Progress")  # Create progress bar

        self.global_steps += 1  # Start from step 1
        last_val_metrics = None  # Store last validation metrics

        timing_raw = defaultdict(float)  # Timing metrics
        batch = None  # Accumulated training batch (may concat under filtering)
        num_prompt_in_batch = 0  # Number of distinct prompts accumulated
        num_gen_batches = 0  # Number of generated sub-batches (for filtering)
        
        # GRPO-specific configuration
        use_global_reward = self.config.algorithm.get("use_global_reward", True)  # Enable global reward
        global_reward_weight = self.config.algorithm.get("global_reward_weight", 1.0)  # Global reward weight (complements token-level)
        
        for epoch in range(self.config.trainer.total_epochs):  # Iterate by epoch
            for batch_dict in self.train_dataloader:  # Iterate training dataloader
                metrics = {}  # Metrics to record this step
                # Prepare batch data
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)  # Build DataProto from raw dict
                # num_gen_batches increases when generation actually happens

                # Handle multi-modal data
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():  # If multi-modal fields present
                    gen_batch = new_batch.pop(  # Pop fields needed for generation to form gen_batch
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(  # Text-only: pop corresponding fields
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps  # Whether this is the final step

                # Training step
                with _timer("step", timing_raw):  # Timing: single training step
                    # Generate sequences
                    with _timer("gen", timing_raw):
                        num_gen_batches += 1  # Generation happens once for accumulated candidates
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)  # Generate with remote actor worker
                    
                    # Compute global rewards (GRPO core)
                    if use_global_reward and self.global_reward_fn:  # If enabled and function provided
                        with _timer("global_reward", timing_raw):  # Timing: global reward
                            # Compute sequence-level global reward
                            global_rewards = self.global_reward_fn(new_batch)  # May return [B] or [B, T]
                            global_rewards = global_rewards.sum(dim=-1) if global_rewards.dim() > 1 else global_rewards  # If [B, T], sum over T
                            new_batch.batch["global_rewards"] = global_rewards  # Store in batch

                    # Prepare batch with stable prompt keys and unique UIDs
                    # 1) Build a stable prompt_key (prefer raw_prompt_ids to identify the same prompt across steps)
                    if "raw_prompt_ids" in gen_batch.non_tensor_batch:
                        def _to_key(x):
                            try:
                                return str(tuple(np.asarray(x).reshape(-1).tolist()))
                            except Exception:
                                return str(x)
                        prompt_keys = np.array([_to_key(x) for x in gen_batch.non_tensor_batch["raw_prompt_ids"]], dtype=object)
                    else:
                        # Fallback: temporary keys if no stable ID (valid only within this step)
                        prompt_keys = np.array([f"auto_{i}" for i in range(len(new_batch.batch))], dtype=object)
                    new_batch.non_tensor_batch["prompt_key"] = prompt_keys

                    # 2) Generate a unique UID per prompt in this step (to group multiple rollouts per prompt)
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], 
                        dtype=object
                    )

                    # 3) After repeat, both uid and prompt_key are repeated and tagged to each trajectory
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, 
                        interleave=True
                    )
                    new_batch = new_batch.union(gen_batch_output)  # Merge generated sequence outputs into batch

                    # Compute rewards
                    with _timer("reward", timing_raw):  # Timing: reward computation
                        # Standard reward model
                        if self.use_rm:  # If external RM (reward model) enabled
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)  # Compute scores via RM
                            new_batch = new_batch.union(reward_tensor)  # Merge back into batch
                        
                        # GRPO: combine token-level and global rewards
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)  # Expect dict from base reward_fn
                            token_rewards = reward_result["reward_tensor"]  # Token-level reward matrix [B, T]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]  # Extra info (e.g., sub-metrics)
                        except Exception:
                            token_rewards = self.reward_fn(new_batch)  # Backward-compatible: tensor only
                            reward_extra_infos_dict = {}
                        
                        # Apply global reward weighting
                        if use_global_reward and "global_rewards" in new_batch.batch:  # If global rewards available
                            global_rewards = new_batch.batch["global_rewards"]  # [B]
                            # Expand to match token-level dimension
                            expanded_global = global_rewards.unsqueeze(-1).expand(  # [B, T]
                                -1, token_rewards.size(1))
                            
                            # Combine token-level and global rewards
                            combined_rewards = (1 - global_reward_weight) * token_rewards + \
                                              global_reward_weight * expanded_global  # Weighted linear fusion
                            new_batch.batch["token_level_rewards"] = combined_rewards  # Write back fused token-level rewards
                            metrics["train/global_reward_weight"] = global_reward_weight  # Record weight
                        else:
                            new_batch.batch["token_level_rewards"] = token_rewards  # Token-level rewards only
                        
                        # Additional reward info
                        if reward_extra_infos_dict:  # If extra info present
                            new_batch.non_tensor_batch.update(  # Merge into non-tensor fields as numpy arrays
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                    # Apply KL penalty if configured
                    if self.config.algorithm.use_kl_in_reward:  # Add KL penalty in reward
                        new_batch, kl_metrics = apply_kl_penalty(  # Apply KL and return new batch + metrics
                            new_batch, 
                            kl_ctrl=self.kl_ctrl_in_reward, 
                            kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)  # Record KL metrics
                    else:
                        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_rewards"]  # Keep rewards (placeholder)

                    # Batch filtering
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        fg_cfg = self.config.algorithm.filter_groups
                        metric_name = fg_cfg.metric

                        # Get per-trajectory metric values (e.g., seq_final_reward/seq_reward)
                        if metric_name == "seq_final_reward":
                            vals_np = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().float().numpy()
                            )
                            new_batch.non_tensor_batch["seq_final_reward"] = vals_np
                        elif metric_name == "seq_reward":
                            vals_np = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().float().numpy()
                            )
                            new_batch.non_tensor_batch["seq_reward"] = vals_np
                        else:
                            vals_np = new_batch.non_tensor_batch[metric_name]
                            if isinstance(vals_np, list):
                                vals_np = np.asarray(vals_np, dtype=float)

                        uids = new_batch.non_tensor_batch["uid"]
                        pkeys = new_batch.non_tensor_batch["prompt_key"]

                        # ---- Aggregate via dicts (avoid bincount weight length mismatch) ----
                        uid2vals = defaultdict(list)
                        uid2pkey = {}
                        for uid, pk, v in zip(uids, pkeys, vals_np):
                            uid2vals[uid].append(float(v))
                            if uid not in uid2pkey:
                                uid2pkey[uid] = pk

                        threshold_persist = getattr(fg_cfg, "threshold_persist", None)
                        threshold_upper = getattr(fg_cfg, "threshold_upper", 1.0)
                        threshold_lower = getattr(fg_cfg, "threshold_lower", None)
                        eps = getattr(fg_cfg, "threshold_eps", 1e-8)

                        kept_uids = []
                        newly_block_keys = set()
                        persist_dropped = 0
                        upper_dropped = 0
                        lower_dropped = 0

                        for uid, vals in uid2vals.items():
                            mean_v = float(np.mean(vals)) if len(vals) > 0 else 0.0
                            pk = uid2pkey[uid]

                            already_blocked = pk in self._fg_persist_block
                            hit_persist = (threshold_persist is not None) and (mean_v >= (threshold_persist - eps))
                            hit_upper = mean_v >= (threshold_upper - eps)
                            hit_lower = (threshold_lower is not None) and (mean_v <= (threshold_lower + eps))

                            if hit_persist and not already_blocked:
                                newly_block_keys.add(pk)

                            drop_due_persist = already_blocked or hit_persist
                            drop_due_upper_only = (not drop_due_persist) and hit_upper
                            drop_due_lower_only = (not drop_due_persist) and (not hit_upper) and hit_lower

                            if drop_due_persist:
                                persist_dropped += 1
                            elif drop_due_upper_only:
                                upper_dropped += 1
                            elif drop_due_lower_only:
                                lower_dropped += 1

                            drop = drop_due_persist or drop_due_upper_only or drop_due_lower_only
                            if not drop:
                                kept_uids.append(uid)

                        # Update persistent blocklist and totals; keep dynamic step-adjustment logic
                        if newly_block_keys:
                            self._fg_persist_block.update(newly_block_keys)
                            self._fg_x1_total_unique = len(self._fg_persist_block)
                            try:
                                delta_steps = int(np.ceil(len(newly_block_keys) / max(1, self.config.data.train_batch_size)))
                                if delta_steps > 0:
                                    self.total_training_steps = max(self.global_steps, self.total_training_steps - delta_steps)
                                    progress_bar.total = max(progress_bar.n, self.total_training_steps)
                                    progress_bar.refresh()
                            except Exception:
                                pass

                        # Select trajectory indices by kept UIDs
                        kept_traj_idxs = [i for i, uid in enumerate(uids) if uid in kept_uids]

                        # Filter batch by mask
                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        # Stats (per prompt)
                        total_prompts_now = len(uid2vals)
                        kept_prompts_now = len(kept_uids)
                        metrics["filter/persist_threshold"] = float(threshold_persist) if threshold_persist is not None else float("nan")
                        metrics["filter/persist_newly_blocklisted"] = len(newly_block_keys)
                        metrics["filter/persist_dropped_in_batch"] = int(persist_dropped)
                        metrics["filter/persist_total_unique_prompts"] = int(self._fg_x1_total_unique)
                        metrics["filter/upper_threshold"] = float(threshold_upper)
                        metrics["filter/upper_dropped"] = int(upper_dropped)
                        if threshold_lower is not None:
                            metrics["filter/lower_threshold"] = float(threshold_lower)
                            metrics["filter/lower_dropped"] = int(lower_dropped)


                        # Prompt accumulation control (do not force-fill to train_batch_size; train even if fewer)
                        num_prompt_in_batch += kept_prompts_now
                        prompt_bsz_cfg = self.config.data.train_batch_size
                        # Prompts used for training in this step = currently kept count
                        used_prompt_bsz = num_prompt_in_batch
                        used_traj_bsz = used_prompt_bsz * self.config.actor_rollout_ref.rollout.n
                        # Truncate to the number of trajectories available this step
                        batch = batch[:used_traj_bsz]
                        # Record actual batch size used
                        metrics["train/used_prompt_batch_size"] = int(used_prompt_bsz)
                        metrics["train/config_train_batch_size"] = int(prompt_bsz_cfg)

                    # Compute response mask
                    batch.batch["response_mask"] = compute_response_mask(batch)  # Mask of response tokens (train only on answers)
                    
                    # Balance batch if configured
                    if self.config.trainer.balance_batch:  # Enable batch balancing by class/length, etc.
                        self._balance_batch(batch, metrics=metrics)  # Rebalance and record metrics
                    
                    # Global token count
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()  # Effective token count per sample
                    
                    # Compute old log probabilities
                    with _timer("old_log_prob", timing_raw):  # Timing: old-policy log prob
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # Actor computes log prob and entropy
                        entropys = old_log_prob.batch["entropys"]  # Entropy tensor
                        response_masks = batch.batch["response_mask"]  # Mask
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode  # Aggregation mode (mean/sum/last)
                        entropy_loss = agg_loss(  # Aggregated masked entropy
                            loss_mat=entropys, 
                            loss_mask=response_masks, 
                            loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}  # Record entropy metric
                        metrics.update(old_log_prob_metrics)  # Merge into overall metrics
                        old_log_prob.batch.pop("entropys")  # Remove entropy from batch (avoid duplication/save space)
                        batch = batch.union(old_log_prob)  # Merge log prob, etc., into main batch
                    
                    # Reference policy computation
                    if self.use_reference_policy:  # Use reference policy (e.g., PPO-ptx/GRPO)
                        with _timer("ref", timing_raw):  # Timing: reference policy
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # Ref policy log prob on same sequences
                            batch = batch.union(ref_log_prob)  # Merge into batch
                    
                    # Value computation
                    if self.use_critic:  # Use value network
                        with _timer("values", timing_raw):  # Timing: value estimation
                            values = self.critic_wg.compute_values(batch)  # Compute V(s) per token
                            batch = batch.union(values)  # Merge into batch
                    
                    # Advantage computation (GRPO specific)
                    with _timer("adv", timing_raw):  # Timing: advantage estimation
                        batch = compute_advantage(  # Advantage (GRPO: normalization/multiple samples)
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        )
                    
                    # Critic update
                    if self.use_critic:  # Use value network
                        with _timer("update_critic", timing_raw):  # Timing: critic update
                            critic_output = self.critic_wg.update_critic(batch)  # Backprop and update critic
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])  # Reduce critic metrics
                        metrics.update(critic_output_metrics)  # Merge into overall metrics
                    
                    # Actor update (after critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:  # Update actor after warmup
                        with _timer("update_actor", timing_raw):  # Timing: actor update
                            actor_output = self.actor_rollout_wg.update_actor(batch)  # Backprop and update actor
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])  # Reduce actor metrics
                        metrics.update(actor_output_metrics)  # Merge into overall metrics
                    
                    # Validation
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):  # Validation at frequency or last step
                        with _timer("testing", timing_raw):  # Timing: validation
                            val_metrics: dict = self._validate()  # Run validation and get metrics
                            if is_last_step:
                                last_val_metrics = val_metrics  # Store last validation result
                        metrics.update(val_metrics)  # Merge validation metrics
                    
                    # Save checkpoint
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):  # Save by frequency or at last step
                        with _timer("save_checkpoint", timing_raw):  # Timing: save
                            self._save_checkpoint()  # Persist model/optimizer, etc.

                # For compute_data_metrics compatibility: alias token_level_rewards to token_level_scores
                if "token_level_rewards" in batch.batch and "token_level_scores" not in batch.batch:  # Metrics expect 'scores'
                    batch.batch["token_level_scores"] = batch.batch["token_level_rewards"]  # Reuse rewards as scores
                    
                # Compute and log metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))  # Data metrics
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))  # Timing metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()  # Query GPU count
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))  # Throughput metrics
                timing_raw = defaultdict(float)  # Reset timing accumulator

                # GRPO-specific metrics
                metrics["train/num_gen_batches"] = num_gen_batches  # Number of generated sub-batches
                if "global_rewards" in batch.batch:  # Global reward present
                    metrics["reward/global_reward_mean"] = batch.batch["global_rewards"].mean().item()  # Mean
                    metrics["reward/global_reward_std"] = batch.batch["global_rewards"].std().item()  # Std
                
                batch = None  # Clear accumulated batch
                num_prompt_in_batch = 0  # Reset prompt counter
                num_gen_batches = 0  # Reset generated-batch counter

                # Log metrics
                logger.log(data=metrics, step=self.global_steps)  # Log all metrics

                # Final validation and exit
                if is_last_step:  # If at last step
                    pprint(f"Final validation metrics: {last_val_metrics}")  # Print final validation metrics
                    progress_bar.close()  # Close progress bar
                    return  # End training

                # Update progress
                progress_bar.update(1)  # Advance progress bar
                self.global_steps += 1  # Increment step
