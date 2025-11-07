# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses (avg token logprob & entropy) with vLLM, a progress bar, and spawn-based multiprocessing.

- Input: Parquet; the column at `config.data.prompt_key` is a chat list ([{role, content}, ...]).
- Output: Parquet; new columns:
    responses: list of strings per row with length `n_samples`.
    logprob:   list of floats per row (mean token logprob).
    entropy:   list of floats per row (approximated via top-k).
- thinking: automatically inject a <think>... system prompt.
- Maximize VRAM: `gpu_memory_utilization≈0.98` (or read `config.rollout.gpu_memory_utilization`).
- Key fix: enforce spawn to avoid "Cannot re-initialize CUDA in forked subprocess".
"""

# Copyright ...
# (Header repeated as above)

import os
import math
import hydra
import numpy as np
import pandas as pd
from pprint import pprint
from omegaconf import OmegaConf

# ---- Important: set spawn to avoid CUDA in forked subprocesses ----
import multiprocessing as mp
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm


def _visible_gpu_count():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis:
        return 1
    return len([x for x in vis.split(",") if x.strip() != ""])

def _apply_thinking_system_message(chat, think_prompt):
    if not chat or not isinstance(chat, list):
        return chat
    sys_msg = {"role": "system", "content": think_prompt}
    if chat and chat[0].get("role") == "system":
        return [sys_msg] + chat
    return [sys_msg] + chat

def _norm_logprob_value(v):
    """Normalize possible vLLM logprob representations (float/object/dict) to a float."""
    try:
        # object has .logprob attribute
        if hasattr(v, "logprob"):
            return float(v.logprob)
        # dict contains "logprob"
        if isinstance(v, dict):
            if "logprob" in v:
                return float(v["logprob"])
            # fallback: try other keys
            for k in ("lp", "value", "score"):
                if k in v:
                    return float(v[k])
        # already a float
        return float(v)
    except Exception:
        return None

def _extract_chosen_token_logprobs(choice, tokenizer):
    """
    Extract per-step logprobs for the chosen tokens (list[float]).
    Prefer `token_logprobs`; otherwise derive from per-step top-k dict using the chosen token string/ID.
    """
    # 1) Prefer: token_logprobs
    if hasattr(choice, "token_logprobs") and choice.token_logprobs is not None:
        vals = []
        for x in choice.token_logprobs:
            if x is None:
                continue
            try:
                vals.append(float(x))
            except Exception:
                lp = _norm_logprob_value(x)
                if lp is not None:
                    vals.append(lp)
        if vals:
            return vals

    # 2) Only top-k dict available: choice.logprobs or choice.top_logprobs
    top_list = getattr(choice, "logprobs", None)
    if top_list is None:
        top_list = getattr(choice, "top_logprobs", None)

    if isinstance(top_list, list) and top_list:
        # Find generated token strings (field names may differ by version)
        tokens = getattr(choice, "tokens", None)
        if tokens is None:
            tokens = getattr(choice, "output_tokens", None)

        if tokens is None:
            # Recover strings from token_ids
            token_ids = getattr(choice, "token_ids", None) or getattr(choice, "output_token_ids", None)
            if token_ids is not None and tokenizer is not None:
                try:
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                except Exception:
                    tokens = None

        vals = []
        for i, step_dict in enumerate(top_list):
            if not isinstance(step_dict, dict):
                continue
            chosen_lp = None
            # If tokens available, use the chosen token's logprob
            if tokens is not None and i < len(tokens) and tokens[i] in step_dict:
                chosen_lp = _norm_logprob_value(step_dict[tokens[i]])
            # If not found, approximate with the max logprob in this step's top-k
            if chosen_lp is None and step_dict:
                best = None
                for _, v in step_dict.items():
                    lp = _norm_logprob_value(v)
                    if lp is None:
                        continue
                    if best is None or lp > best:
                        best = lp
                chosen_lp = best
            if chosen_lp is not None:
                vals.append(float(chosen_lp))
        if vals:
            return vals

    # 3) Fallback: token_scores (some versions)
    if hasattr(choice, "token_scores") and isinstance(choice.token_scores, list):
        vals = []
        for v in choice.token_scores:
            lp = _norm_logprob_value(v)
            if lp is not None:
                vals.append(lp)
        return vals

    return []

def _normalize_top_logprobs_for_entropy(choice):
    """
    Normalize the top-k list to list[dict[str, float]] for entropy approximation.
    """
    top_list = getattr(choice, "top_logprobs", None)
    if top_list is None:
        top_list = getattr(choice, "logprobs", None)
    if not isinstance(top_list, list):
        return None
    norm = []
    for step in top_list:
        if not isinstance(step, dict):
            norm.append({})
            continue
        d = {}
        for tok, v in step.items():
            lp = _norm_logprob_value(v)
            if lp is not None:
                d[str(tok)] = lp
        norm.append(d)
    return norm

def _approx_entropy_from_top_logprobs(top_logprobs_list):
    """
    Approximate entropy using per-step top-k:
    H ≈ -∑ p_i log p_i - (1 - ∑ p_i) * log(max(eps, 1 - ∑ p_i))
    """
    if not top_logprobs_list:
        return float("nan")
    entropies, eps = [], 1e-12
    for step in top_logprobs_list:
        if not step:
            entropies.append(float("nan"))
            continue
        probs = [math.exp(lp) for lp in step.values()]
        p_sum = sum(probs)
        h_top = -sum(p * math.log(max(p, eps)) for p in probs)
        p_tail = max(0.0, 1.0 - p_sum)
        h_tail = -p_tail * math.log(max(p_tail, eps))
        entropies.append(h_top + h_tail)
    entropies = [e for e in entropies if not (isinstance(e, float) and math.isnan(e))]
    return float(np.mean(entropies)) if entropies else float("nan")

@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)

def run_generation(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    model_path = config.model.path
    trust_remote_code = config.data.get("trust_remote_code", False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]

    n_samples = int(config.data.n_samples)
    assert n_samples >= 1
    if float(getattr(config.rollout, "temperature", 1.0)) == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."

    prompt_max_len = int(getattr(config.rollout, "prompt_length", 4096))
    max_new_tokens = int(getattr(config.rollout, "response_length",
                          getattr(config.rollout, "max_new_tokens", 512)))

    enable_thinking = bool(getattr(config.data, "enable_thinking", True))
    think_prompt = getattr(
        config.data,
        "thinking_system_prompt",
        "You are in thinking mode. Think step by step inside <think>...</think> before the final answer."
    )

    # vLLM top_logprobs cap is typically 20
    requested_top_logprobs = int(getattr(config.data, "top_logprobs_k", 20))
    MAX_LOGPROBS_CAP = int(os.getenv("VLLM_MAX_LOGPROBS_CAP", "20"))
    effective_top_logprobs = max(0, min(requested_top_logprobs, MAX_LOGPROBS_CAP))

    tp_cfg = int(getattr(config.rollout, "tensor_model_parallel_size", 0))
    tensor_parallel_size = tp_cfg if tp_cfg > 0 else _visible_gpu_count()
    gpu_mem_util = float(getattr(config.rollout, "gpu_memory_utilization",
                                 getattr(config.data, "gpu_mem_utilization", 0.98)))

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        max_logprobs=MAX_LOGPROBS_CAP,  # supported in some versions; harmless if ignored
    )

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=float(getattr(config.rollout, "temperature", 0.7)),
        top_p=float(getattr(config.rollout, "top_p", 1.0)),
        top_k=int(getattr(config.rollout, "top_k", -1)),
        max_tokens=max_new_tokens,
        repetition_penalty=float(getattr(config.rollout, "repetition_penalty", 1.0)),
        stop=getattr(config.rollout, "stop", None),
        logprobs=effective_top_logprobs,
    )

    # prompts
    prompts = []
    for chat in chat_lst:
        chat_ = _apply_thinking_system_message(chat, think_prompt) if enable_thinking else chat
        prompt_text = tokenizer.apply_chat_template(
            chat_,
            add_generation_prompt=True,
            padding=False,
            truncation=True,
            max_length=prompt_max_len,
            tokenize=False,
        )
        prompts.append(prompt_text)

    # batching
    bsz = int(config.data.batch_size)
    num_batch = (len(prompts) + bsz - 1) // bsz

    responses_all = [[] for _ in range(n_samples)]
    logprob_all = [[] for _ in range(n_samples)]
    entropy_all = [[] for _ in range(n_samples)]

    total_samples_all = len(prompts) * n_samples
    pbar = tqdm(total=total_samples_all, desc="vLLM generation", unit="sample", dynamic_ncols=True)

    for bi in range(num_batch):
        batch_prompts = prompts[bi * bsz : (bi + 1) * bsz]
        request_outputs = llm.generate(batch_prompts, sampling_params)

        for ro in request_outputs:
            for s_idx in range(n_samples):
                if s_idx < len(ro.outputs):
                    choice = ro.outputs[s_idx]
                    text = choice.text or ""

                    # Key change: robust extraction of per-step chosen-token logprobs
                    token_lp_list = _extract_chosen_token_logprobs(choice, tokenizer)
                    mean_lp = float(np.mean(token_lp_list)) if token_lp_list else float("nan")

                    # Approximate entropy (normalize top-k first, then compute)
                    top_norm = _normalize_top_logprobs_for_entropy(choice)
                    mean_ent = _approx_entropy_from_top_logprobs(top_norm) if isinstance(top_norm, list) else float("nan")

                    responses_all[s_idx].append(text)
                    logprob_all[s_idx].append(mean_lp)
                    entropy_all[s_idx].append(mean_ent)
                else:
                    responses_all[s_idx].append("")
                    logprob_all[s_idx].append(float("nan"))
                    entropy_all[s_idx].append(float("nan"))
            pbar.update(n_samples)

    pbar.close()

    # (n_samples, n_data) -> (n_data, n_samples)
    responses_all = np.array(responses_all, dtype=object).T.tolist()
    logprob_all = np.array(logprob_all, dtype=object).T.tolist()
    entropy_all = np.array(entropy_all, dtype=object).T.tolist()

    dataset["responses"] = responses_all
    dataset["logprob"]  = logprob_all
    dataset["entropy"]  = entropy_all

    out_path = config.data.output_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dataset.to_parquet(out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
