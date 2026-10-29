import json
import time
import os
import numpy as np
from tabulate import tabulate

from src.matcher import BaselineMatcher, SemanticMatcher, MatchDecision


def load_dataset(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_benchmark():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_products.json")
    products = load_dataset(data_path)
    prod_map = {p["id"]: p for p in products}

    # Ground-truth annotated pairs
    test_pairs = [
        # (ID_A, ID_B, Ground_Truth_Label)
        ("PROD_001", "PROD_002", MatchDecision.EXACT_DUPLICATE),      # Sony Headphones Black (duplicate wording)
        ("PROD_001", "PROD_003", MatchDecision.POTENTIAL_VARIATION),  # Sony Headphones (Black vs Silver)
        ("PROD_004", "PROD_005", MatchDecision.POTENTIAL_VARIATION),  # iPhone 15 Pro (128GB vs 256GB)
        ("PROD_006", "PROD_007", MatchDecision.EXACT_DUPLICATE),      # Hydro Flask 32oz (duplicate wording/units)
        ("PROD_001", "PROD_008", MatchDecision.DISTINCT),             # Headphones vs Power Bank
        ("PROD_004", "PROD_008", MatchDecision.DISTINCT),             # iPhone vs Power Bank
    ]

    print("=================================================================")
    print("         AMAZON CATALOG DEDUPLICATION EVALUATION BENCHMARK        ")
    print("=================================================================\n")

    baseline = BaselineMatcher()
    all_titles = [p["title"] for p in products]
    baseline.fit(all_titles)

    semantic = SemanticMatcher()

    baseline_latencies = []
    semantic_latencies = []

    baseline_correct = 0
    semantic_correct = 0

    results_table = []

    for id_a, id_b, ground_truth in test_pairs:
        item_a = prod_map[id_a]
        item_b = prod_map[id_b]

        # 1. Benchmark Baseline (TF-IDF)
        t0 = time.perf_counter()
        sim_base = baseline.compute_similarity(item_a["title"], item_b["title"])
        baseline_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Baseline naive classification rule
        if sim_base >= 0.70:
            pred_base = MatchDecision.EXACT_DUPLICATE
        elif sim_base >= 0.40:
            pred_base = MatchDecision.POTENTIAL_VARIATION
        else:
            pred_base = MatchDecision.DISTINCT

        if pred_base == ground_truth:
            baseline_correct += 1

        # 2. Benchmark Semantic Matcher
        t0 = time.perf_counter()
        pred_sem, sim_sem, meta = semantic.classify_pair(item_a, item_b)
        semantic_latencies.append((time.perf_counter() - t0) * 1000.0)

        if pred_sem == ground_truth:
            semantic_correct += 1

        results_table.append([
            f"{id_a} & {id_b}",
            ground_truth.value,
            pred_base.value,
            f"{sim_base:.2f}",
            pred_sem.value,
            f"{sim_sem:.2f}",
            "PASS" if pred_sem == ground_truth else "FAIL"
        ])

    headers = ["Pair", "Ground Truth", "TF-IDF Pred", "TF-IDF Sim", "Semantic Pred", "Semantic Sim", "Correct"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    print("\n")

    summary_table = [
        ["Model", "Accuracy", "p50 Latency (ms)", "p95 Latency (ms)"],
        [
            "Baseline (TF-IDF)",
            f"{(baseline_correct / len(test_pairs)) * 100:.1f}%",
            f"{np.percentile(baseline_latencies, 50):.2f}",
            f"{np.percentile(baseline_latencies, 95):.2f}"
        ],
        [
            "Semantic (MiniLM + Norm)",
            f"{(semantic_correct / len(test_pairs)) * 100:.1f}%",
            f"{np.percentile(semantic_latencies, 50):.2f}",
            f"{np.percentile(semantic_latencies, 95):.2f}"
        ]
    ]

    print(tabulate(summary_table, headers="firstrow", tablefmt="github"))
    print("\n[INFO] Evaluation completed successfully.")


if __name__ == "__main__":
    run_benchmark()
