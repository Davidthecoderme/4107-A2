import pytrec_eval

# 1. Load QREL file
qrel = {}
with open("formated-test.tsv", "r") as f:
    for line in f:
        qid, _, docid, rel = line.strip().split()
        qrel.setdefault(qid, {})[docid] = int(rel)

# 2. Load system output (filtered results)
run = {}
with open("Results_bert.txt", "r") as f: 
    for line in f:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        qid, _, docid, _, score, _ = parts
        run.setdefault(qid, {})[docid] = float(score)

# 3. Evaluate with pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'recip_rank', 'P_10'})
results = evaluator.evaluate(run)

# 4. Print average results
average_metrics = {metric: 0.0 for metric in ['map', 'ndcg', 'recip_rank', 'P_10']}
for query_id in results:
    for metric in average_metrics:
        average_metrics[metric] += results[query_id][metric]

query_count = len(results)
for metric in average_metrics:
    average_metrics[metric] /= query_count
    print(f'{metric}: {average_metrics[metric]:.4f}')
