from reward import compute_score, extract_answer
import pandas as pd
import numpy as np
from collections import Counter

scores = {}
results = {}
df = pd.read_parquet("STaR/checkpoints/STaR/STaR-Qwen3-0.6B-Stage1-Stage2/global_step_190/actor/huggingface/STaR-eval-8.parquet")

def consistency_confidence_fusion(responses, logprobs, entropies):
    # Step 1: find all possible answers
    answer_responses = {}
    for response, logprob, entropy in zip(responses, logprobs, entropies):
            
        answers = extract_answer(response)
        if answers is None:
            continue
            
        answer_key = tuple(answers)
        if answer_key not in answer_responses:
            answer_responses[answer_key] = []
        
        # compute combined confidence
        confidence = np.exp(logprob) * (1 - entropy)
        answer_responses[answer_key].append({
            'response': response,
            'confidence': confidence
        })
    
    if not answer_responses:
        return None
    
    # Collect all metrics for normalization
    all_consistencies = []
    all_avg_confidences = []
    all_max_confidences = []
    
    for answer_key, responses_list in answer_responses.items():
        consistency = len(responses_list)
        avg_confidence = np.mean([r['confidence'] for r in responses_list])
        max_confidence = max([r['confidence'] for r in responses_list])
        
        all_consistencies.append(consistency)
        all_avg_confidences.append(avg_confidence)
        all_max_confidences.append(max_confidence)
    
    # Compute maximum values for normalization
    max_consistency = max(all_consistencies)
    max_avg_confidence = max(all_avg_confidences)
    max_max_confidence = max(all_max_confidences)
    
    # Compute normalized scores for each answer
    answer_scores = {}
    for answer_key, responses_list in answer_responses.items():
        consistency = len(responses_list)
        avg_confidence = np.mean([r['confidence'] for r in responses_list])
        max_confidence = max([r['confidence'] for r in responses_list])
        
        # Normalize each metric
        norm_consistency = consistency / max_consistency
        norm_avg_confidence = avg_confidence / max_avg_confidence
        norm_max_confidence = max_confidence / max_max_confidence

        # Combined score (from normalized metrics)
        answer_scores[answer_key] = {
            'score': norm_consistency * 0.25 + norm_avg_confidence * 0.2 + norm_max_confidence * 0.55,
            'best_response': max(responses_list, key=lambda x: x['confidence'])['response']
        }
    
    # Choose the answer with highest combined score
    best_answer = max(answer_scores.items(), key=lambda x: x[1]['score'])
    return best_answer[1]['best_response']

for i in range(len(df)):
    responses = df.iloc[i]["responses"]
    logprobs = df.iloc[i]["logprob"]
    entropies = df.iloc[i]["entropy"]
    ground_truth = df.iloc[i]["reward_model"]["ground_truth"]
    data_source = df.iloc[i]["data_source"]
        
    if not (len(responses) == len(logprobs) == len(entropies)):
        continue
    
    # Use consistency-confidence fusion method to select the best response
    selected_response = consistency_confidence_fusion(responses, logprobs, entropies)
    
    # If no valid response, skip this row
    if selected_response is None:
        continue
    
    # Compute score using the selected response
    score = compute_score(None, selected_response, ground_truth, None)
    
    if data_source not in scores:
        scores[data_source] = []
    scores[data_source].append(score["accurate_score"])

for data_source, score_list in scores.items():
    results[data_source] = np.mean(score_list)

print(results)