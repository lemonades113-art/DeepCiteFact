import json

f = open('rollout_data_step_1.jsonl', encoding='utf-8')
lines = [json.loads(l) for l in f][:3]

for i, x in enumerate(lines):
    print(f"=== Sample {i+1} ===")
    print(f"query: {x['query'][:150]}")
    print(f"fact_reward: {x['fact_reward']:.4f}")
    print(f"citation_reward: {x['citation_reward']}")
    print(f"search_reward: {x['search_reward']}")
    print(f"format_reward: {x['format_reward']}")
    print(f"search_num: {x['search_num']}")
    print()
    resp = x['response']
    print(f"response (first 2000 chars):")
    print(resp[:2000])
    print("\n" + "="*80 + "\n")
