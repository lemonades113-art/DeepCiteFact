import openai
import os
import math
import concurrent.futures
import random
import time
import re
import requests
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from verl.workers.reward_manager.utils.prompts import claim_request_prompt_template, claim_check_prompt_template

# 加载server
CHECK_SERVER = os.environ.get('CHECK_SERVER', 'https://api.siliconflow.cn')
CHECK_SERVER_PATH = os.environ.get('CHECK_SERVER_PATH', 'Qwen/Qwen2.5-32B-Instruct')

CLAIM_SERVER = os.environ.get('CLAIM_SERVER', 'https://api.siliconflow.cn')
CLAIM_SERVER_PATH = os.environ.get('CLAIM_SERVER_PATH', 'Qwen/Qwen2.5-32B-Instruct')

SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY', 'sk-iztmlrhucomamftbaowufnonkkjvcnezdsrihgcnmlctlbdd')
    

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*)</answer>'
    
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) <= 0:
        return ""

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def request_claims(prompt):
    client = openai.Client(base_url=f"{CLAIM_SERVER}/v1", api_key=SILICONFLOW_API_KEY)
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=CLAIM_SERVER_PATH,
                messages=[
                    {"role": "system", "content": "You must respond ONLY with bullet points in the format '* claim'. No other text. If no verifiable claims exist, respond with 'no verifiable objective claims'."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()

        except:
            time.sleep(1)
            continue
            
    return ""


def parse_claims(response):
    if not response or "no verifiable objective claims" in response.lower():
        return []
    try:
        import re as _re
        claims = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('* '):
                claims.append(line[2:])
            elif _re.match(r'^\d+\.\s+', line):
                claims.append(_re.sub(r'^\d+\.\s+', '', line))
            elif line.startswith('- '):
                claims.append(line[2:])
        if not claims:
            sentences = _re.split(r'(?<=[.!?])\s+', response.strip())
            claims = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
        return list(dict.fromkeys(claims))  # drop duplicate
    except Exception as e:
        print(f"error {e}")
        return []


def request_check(claim):
    prompt = claim_check_prompt_template.format(claim=claim)
    client = openai.Client(base_url=f"{CHECK_SERVER}/v1", api_key=SILICONFLOW_API_KEY)
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=CHECK_SERVER_PATH,
                messages=[
                    {"role": "system", "content": "You must respond with exactly one word: True or False."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                top_p=0.1,
                max_tokens=5
            )
            text = response.choices[0].message.content.strip()
            first_word = text.split()[0].rstrip('.,') if text else ''
            if first_word.lower() == 'true':
                return 1.0
            elif first_word.lower() == 'false':
                return 0.0
        except:
            time.sleep(1)
            continue
            
    return 0.5  # uncertain fallback


def compute_single_reward(response):
    # 1. 使用Qwen2.5-32B-Instruct分解
    prompt = claim_request_prompt_template.format(response=response)
    claim_response = request_claims(prompt)
    claims = parse_claims(claim_response)
    # 自身验证
    max_workers = 32
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request_check, claim) for claim in claims]
        scores = [future.result() for future in futures]
    
    if len(scores) == 0:
        return 0.0  # no answer or no verifiable claims → no reward
    else:
        return np.mean(scores)
    

def compute_fact_rewards(responses):
    responses = [extract_solution(r) for r in responses]
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating fact rewards, number of tasks {len(responses)}, number of concurrent workers {max_workers}")
        futures = [executor.submit(compute_single_reward, response) for response in responses]
        rewards = [future.result() for future in tqdm(futures)]
    
    return rewards
    