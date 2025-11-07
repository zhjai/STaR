'''
reward for TQA task: WTQ, HiTab
'''

import re
import json
import string

PATTERN = re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>$', re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
STRICT_ANSWER_PATTERN = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\])\s*\})\s*```')
ANSWER_PATTERN_1 = re.compile(r'```json\s*(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})\s*```')
ANSWER_PATTERN_2 = re.compile(r'(\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\})')
NUMBER_PATTERN = re.compile(r'^[+-]?(?:(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def normalize_string(s):
    if isinstance(s, str):
        # 移除非字母数字字符（保留数字和字母）
        s = re.sub(r'[^a-zA-Z0-9]', '', s)
        return s.lower()
    return s

def parse_json(answer):
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or "answer" not in data:
            return None
        if isinstance(data["answer"], list):
            return data["answer"]
        else:
            return [data["answer"]]
    except json.JSONDecodeError:
        return None

def extract_answer_pattern(predict_str):
    answer_match = ANSWER_PATTERN_1.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        return parse_json(answer)
    
    answer_match = ANSWER_PATTERN_2.search(predict_str)
    if answer_match is not None:
        answer = answer_match.group(1).strip()
        return parse_json(answer)
    
    return None

def extract_answer(predict_str):
    answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str)
    if answer_block_match is not None:
        answer = extract_answer_pattern(answer_block_match.group(1))
        if answer is not None:
            return answer

    answer = extract_answer_pattern(predict_str)
    if answer is not None:
        return answer
    
    return None

def normalize_answer(answer):
    normalized_answer = []
    for x in answer:
        if isinstance(x, int) or isinstance(x, float):
            normalized_answer.append(float(x))
        elif isinstance(x, str):
            if NUMBER_PATTERN.match(x):
                try:
                    normalized_answer.append(float(x.replace(',', '')))
                except ValueError:
                    normalized_answer.append(normalize_string(x))
            else:
                normalized_answer.append(normalize_string(x))
        else:
            return []
    return normalized_answer

# for instruct model
def format_check(predict_str):
    if PATTERN.fullmatch(predict_str):
        for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
            if predict_str.count(tag) != 1:
                return False
        answer_block_match = ANSWER_BLOCK_PATTERN.search(predict_str).group(1)
        answer_match = STRICT_ANSWER_PATTERN.search(answer_block_match)
        if answer_match is not None:
            answer = answer_match.group(1).strip()
            final_answer = parse_json(answer)
            if final_answer is None:
                return False
            return True

    return False

def compute_score(data_source, solution_str, ground_truth, extra_info):
    predict_str = solution_str
    answer = extract_answer(predict_str)
    if answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "partial_score": 0.0,
            "accurate_score": 0.0,
        }

    normalized_answer = normalize_answer(answer)
    if len(normalized_answer) == 0 or len(normalized_answer) > 100: 
        return {
            "score": 0.0,
            "format_score": 0.0,
            "partial_score": 0.0,
            "accurate_score": 0.0,
        }
    normalized_ground_truth = normalize_answer(ground_truth)

    format_score = 1.0 if format_check(predict_str) else 0.0
    partial_score = 0.0
    accurate_score = 0.0

    # 检查部分匹配：只要有一个元素匹配就得分
    for a in normalized_answer:
        for b in normalized_ground_truth:
            # 数字比较
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) < 1e-2:
                    partial_score = 1.0
                    break
            # 字符串比较（已规范化）
            elif isinstance(a, str) and isinstance(b, str):
                if a == b:
                    partial_score = 1.0
                    break
        if partial_score == 1.0:
            break

    # 检查完全匹配：忽略顺序，只要元素集合相同就得分
    if len(normalized_answer) == len(normalized_ground_truth):
        # 创建答案和真实值的集合（考虑浮点数精度）
        answer_set = set()
        truth_set = set()
        
        for a in normalized_answer:
            if isinstance(a, float):
                # 将浮点数转换为字符串，保留两位小数，以避免浮点数精度问题
                answer_set.add(f"{a:.2f}")
            else:
                answer_set.add(str(a))
                
        for b in normalized_ground_truth:
            if isinstance(b, float):
                truth_set.add(f"{b:.2f}")
            else:
                truth_set.add(str(b))
                
        if answer_set == truth_set:
            accurate_score = 1.0

    total_score = 0.2 * format_score + 0.3 * partial_score + 0.5 * accurate_score

    return {
        "score": total_score,
        "format_score": format_score,
        "partial_score": partial_score,
        "accurate_score": accurate_score,
    }