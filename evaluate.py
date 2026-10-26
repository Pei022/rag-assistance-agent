import csv
import time
import logging
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)

CATEGORIES = ["simple", "multi_hop", "no_answer", "ambiguous"]


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_gold(gold_str):
    """把 '3,7' 解析成 {3,7}；空字符串返回空集合。"""
    return {int(x) for x in gold_str.split(",") if x.strip().isdigit()}


def hit_at_k(hits, gold_ids):
    """gold 为空（no_answer 类）返回 None，不计入 Hit@K 汇总。"""
    if not gold_ids:
        return None
    return any(h["chunk_id"] in gold_ids for h in hits)


def run_eval(eval_path, out_path):
    pipeline = RAGPipeline()
    rows = load_eval(eval_path)
    results = []

    for r in rows:
        gold_ids = parse_gold(r.get("gold_chunk_ids", ""))
        should_refuse = r.get("should_refuse", "false").lower() == "true"

        t0 = time.time()
        try:
            out = pipeline.generate(r["question"])
            latency_ms = int((time.time() - t0) * 1000)
            answer = out["answer"]
            hits = out["hits"]
            citations = ";".join(
                f"{c['source']}#{c['chunk_id']}" for c in out["citations"]
            )
        except Exception as e:
            logging.error(f"{r['id']} failed: {e}")
            latency_ms, answer, hits, citations = -1, "", [], ""

        hit = hit_at_k(hits, gold_ids)

        # 简单的拒答判断：回答里含"我不知道"或"无法回答"视为拒答
        refused = any(k in answer for k in ["我不知道", "无法回答", "未提及", "没有相关"])

        results.append({
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "gold_answer": r.get("gold_answer", ""),
            "answer": answer,
            "hit_at_k": hit,
            "should_refuse": should_refuse,
            "refused": refused,
            "refuse_correct": (refused == should_refuse),
            "latency_ms": latency_ms,
            "citations": citations,
        })
        logging.info(f"done {r['id']} hit={hit} refused={refused} {latency_ms}ms")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print_summary(results)
    return results


def print_summary(results):
    """按类别汇总 Hit@K 和拒答准确率。"""
    for cat in CATEGORIES:
        subset = [r for r in results if r["category"] == cat]
        if not subset:
            continue
        hits = [r["hit_at_k"] for r in subset if r["hit_at_k"] is not None]
        hit_str = f"Hit@K={sum(hits)/len(hits):.2f}" if hits else "Hit@K=N/A"
        refuse_acc = sum(r["refuse_correct"] for r in subset) / len(subset)
        print(f"[{cat}] n={len(subset)} {hit_str} 拒答准确率={refuse_acc:.2f}")

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    if latencies:
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95) - 1]
        print(f"[时延] avg={sum(latencies)/len(latencies):.0f}ms p95={p95}ms")

if __name__ == "__main__":
    run_eval("data/eval_set.csv", "results/eval_baseline.csv")