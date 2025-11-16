"""
Measurement Script: Compare Execution Time vs Confidence Threshold
Tests confidence thresholds from 0.4 to 0.8 and measures execution time
for association rule generation.
"""

from pyspark.sql import SparkSession
from pyspark import StorageLevel
from itertools import combinations
import time
import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark with optimized configuration
import multiprocessing

num_cores = multiprocessing.cpu_count()

spark = (
    SparkSession.builder.appName("ConfidenceThresholdMeasurement")
    .config("spark.sql.shuffle.partitions", str(num_cores * 4))
    .config("spark.default.parallelism", str(num_cores * 2))
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "2g")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.2")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.rdd.compress", "true")
    .master("local[*]")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Configuration
min_support_threshold = 1000
confidence_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

print("=" * 80)
print("CONFIDENCE THRESHOLD vs EXECUTION TIME MEASUREMENT")
print("=" * 80)
print(f"Min Support: {min_support_threshold}")
print(f"Confidence Thresholds: {confidence_thresholds}")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA (Done once, reused for all confidence thresholds)
# ============================================================================
print("\n[Step 1] Loading and preprocessing data...")
start = time.time()

data_rdd = sc.textFile("hw2/data/data.dat")

# Convert to sets for O(1) lookup and cache
transactions_rdd = (
    data_rdd.map(lambda line: line.strip().split())
    .filter(lambda txn: len(txn) > 0)
    .persist(StorageLevel.MEMORY_AND_DISK)
)

num_transactions = transactions_rdd.count()
print(f"  ✓ Loaded {num_transactions:,} transactions in {time.time()-start:.2f}s")

# ============================================================================
# STEP 2: FIND FREQUENT ITEMSETS (Done once, reused for all confidence thresholds)
# ============================================================================
print("\n[Step 2] Mining frequent itemsets (L1, L2, L3)...")
start_apriori = time.time()

# Find frequent single items (L1)
single_item_counts = (
    transactions_rdd.flatMap(lambda txn: txn)
    .map(lambda item: (item, 1))
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support_threshold)
)

frequent_single_dict = dict(single_item_counts.collect())
print(f"  ✓ Found {len(frequent_single_dict):,} frequent single items")

# Find frequent pairs (L2)
frequent_items_set = set(frequent_single_dict.keys())
frequent_items_bc = sc.broadcast(frequent_items_set)


def generate_pairs(txn):
    """Generate pairs from transaction"""
    items = sorted([item for item in txn if item in frequent_items_bc.value])
    if len(items) < 2:
        return []
    return [
        ((items[i], items[j]), 1)
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


double_itemset_counts = (
    transactions_rdd.flatMap(generate_pairs)
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support_threshold)
)

frequent_double_dict = dict(double_itemset_counts.collect())
print(f"  ✓ Found {len(frequent_double_dict):,} frequent pairs")


# Find frequent triples (L3)
def generate_candidate_triples(frequent_pairs):
    """Generate candidate triples where all sub-pairs are frequent"""
    items = set()
    for pair in frequent_pairs:
        items.update(pair)
    items = sorted(items)

    candidates = []
    for combo in combinations(items, 3):
        a, b, c = combo
        if (
            (a, b) in frequent_pairs
            and (a, c) in frequent_pairs
            and (b, c) in frequent_pairs
        ):
            candidates.append(combo)

    return candidates


candidate_triples = generate_candidate_triples(set(frequent_double_dict.keys()))
candidate_triples_bc = sc.broadcast(set(candidate_triples))


def generate_triples(txn):
    """Generate triples from transaction"""
    items = sorted(txn)
    if len(items) < 3:
        return []
    candidates_set = candidate_triples_bc.value
    triples = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for k in range(j + 1, len(items)):
                triple = (items[i], items[j], items[k])
                if triple in candidates_set:
                    triples.append((triple, 1))
    return triples


triple_itemset_counts = (
    transactions_rdd.flatMap(generate_triples)
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support_threshold)
)

frequent_triple_dict = dict(triple_itemset_counts.collect())
print(f"  ✓ Found {len(frequent_triple_dict):,} frequent triples")

# Create unified support map
unified_support_map = {}
for item, support in frequent_single_dict.items():
    unified_support_map[(item,)] = support
unified_support_map.update(frequent_double_dict)
unified_support_map.update(frequent_triple_dict)

apriori_time = time.time() - start_apriori
print(f"  ✓ A-Priori algorithm completed in {apriori_time:.2f}s")

# ============================================================================
# STEP 3: GENERATE ASSOCIATION RULES FOR EACH CONFIDENCE THRESHOLD
# ============================================================================
print("\n[Step 3] Generating association rules for each confidence threshold...")
print("=" * 80)

# Store results
results = []


def generate_rules_from_itemset(itemset_and_support, support_map_bc, min_confidence):
    """Generate all valid association rules from an itemset"""
    itemset, support = itemset_and_support
    rules = []

    n = len(itemset)
    if n < 2:
        return []

    items = list(itemset)
    support_map = support_map_bc.value

    # Generate all possible rule splits
    for i in range(1, n):
        for antecedent_items in combinations(items, i):
            antecedent = tuple(sorted(antecedent_items))
            consequent = tuple(sorted(set(items) - set(antecedent)))

            # Get antecedent support
            ant_support = support_map.get(antecedent, 0)

            if ant_support == 0:
                continue

            # Calculate confidence
            confidence = support / ant_support

            if confidence >= min_confidence:
                rules.append((antecedent, consequent, confidence, support, ant_support))

    return rules


# Broadcast support map once (reused for all confidence thresholds)
support_map_bc = sc.broadcast(unified_support_map)

# Combine pairs and triples for rule generation
all_frequent_itemsets = list(frequent_double_dict.items()) + list(
    frequent_triple_dict.items()
)

if not all_frequent_itemsets:
    print("No frequent itemsets found. Cannot generate rules.")
    spark.stop()
    exit()

# Test each confidence threshold
for min_confidence in confidence_thresholds:
    print(f"\nTesting confidence threshold: {min_confidence}")
    print("-" * 80)

    start_time = time.time()

    # Generate rules in parallel
    rules_rdd = sc.parallelize(all_frequent_itemsets)
    all_rules = rules_rdd.flatMap(
        lambda x: generate_rules_from_itemset(x, support_map_bc, min_confidence)
    ).collect()

    # Sort by confidence
    all_rules.sort(key=lambda x: x[2], reverse=True)

    execution_time = time.time() - start_time

    # Store results
    results.append(
        {
            "confidence": min_confidence,
            "execution_time": execution_time,
            "num_rules": len(all_rules),
            "apriori_time": apriori_time,
            "total_time": apriori_time + execution_time,
        }
    )

    print(f"  Execution time: {execution_time:.4f}s")
    print(f"  Number of rules generated: {len(all_rules):,}")

    # Show top 5 rules
    if all_rules:
        print(f"  Top 5 rules (by confidence):")
        for i, (antecedent, consequent, confidence, support, ant_support) in enumerate(
            all_rules[:5], 1
        ):
            print(
                f"    {i}. {antecedent} -> {consequent} | confidence: {confidence:.4f}"
            )

# ============================================================================
# DISPLAY SUMMARY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: CONFIDENCE THRESHOLD vs EXECUTION TIME")
print("=" * 80)
print(
    f"{'Confidence':<12} {'Rules':<10} {'Rule Gen Time (s)':<18} {'Total Time (s)':<15} {'Speedup':<10}"
)
print("-" * 80)

# Calculate speedup relative to lowest confidence (0.4)
baseline_time = results[0]["execution_time"] if results else 1.0

for result in results:
    speedup = (
        baseline_time / result["execution_time"] if result["execution_time"] > 0 else 0
    )
    print(
        f"{result['confidence']:<12.1f} "
        f"{result['num_rules']:<10,} "
        f"{result['execution_time']:<18.4f} "
        f"{result['total_time']:<15.4f} "
        f"{speedup:<10.2f}x"
    )

print("=" * 80)
print(f"\nA-Priori algorithm time (one-time cost): {apriori_time:.4f}s")
print(f"Note: A-Priori time is included in Total Time for each confidence threshold")

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
csv_filename = "hw2/confidence_time_measurement.csv"
print(f"\nSaving results to {csv_filename}...")

with open(csv_filename, "w", newline="") as csvfile:
    fieldnames = [
        "confidence",
        "num_rules",
        "rule_generation_time",
        "apriori_time",
        "total_time",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "confidence": result["confidence"],
                "num_rules": result["num_rules"],
                "rule_generation_time": result["execution_time"],
                "apriori_time": result["apriori_time"],
                "total_time": result["total_time"],
            }
        )

print(f"  ✓ Results saved to {csv_filename}")

# ============================================================================
# ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if len(results) > 1:
    # Calculate time differences
    print("\nTime differences between consecutive thresholds:")
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        time_diff = prev["execution_time"] - curr["execution_time"]
        rule_diff = prev["num_rules"] - curr["num_rules"]
        print(
            f"  {prev['confidence']:.1f} -> {curr['confidence']:.1f}: "
            f"Time: {time_diff:+.4f}s, Rules: {rule_diff:+,}"
        )

    # Find fastest and slowest
    fastest = min(results, key=lambda x: x["execution_time"])
    slowest = max(results, key=lambda x: x["execution_time"])
    print(
        f"\nFastest: confidence={fastest['confidence']:.1f} ({fastest['execution_time']:.4f}s)"
    )
    print(
        f"Slowest: confidence={slowest['confidence']:.1f} ({slowest['execution_time']:.4f}s)"
    )

    # Correlation analysis
    print("\nObservations:")
    print("  - Higher confidence thresholds filter out more rules")
    print("  - Execution time typically decreases with higher confidence")
    print("  - This is because fewer rules need to be generated and validated")

print("=" * 80)

# ============================================================================
# CREATE GRAPHS
# ============================================================================
print("\n[Step 4] Generating graphs...")

# Extract data for plotting
confidences = [r["confidence"] for r in results]
execution_times = [r["execution_time"] for r in results]
num_rules = [r["num_rules"] for r in results]
total_times = [r["total_time"] for r in results]

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Graph 1: Confidence Threshold vs Execution Time
ax1 = plt.subplot(2, 2, 1)
ax1.plot(
    confidences, execution_times, marker="o", linewidth=2, markersize=8, color="#2E86AB"
)
ax1.fill_between(confidences, execution_times, alpha=0.3, color="#2E86AB")
ax1.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
ax1.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight="bold")
ax1.set_title(
    "Confidence Threshold vs Execution Time\n(Rule Generation Only)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_xticks(confidences)
for i, (conf, time) in enumerate(zip(confidences, execution_times)):
    ax1.annotate(
        f"{time:.4f}s",
        (conf, time),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )

# Graph 2: Confidence Threshold vs Number of Rules
ax2 = plt.subplot(2, 2, 2)
ax2.plot(confidences, num_rules, marker="s", linewidth=2, markersize=8, color="#A23B72")
ax2.fill_between(confidences, num_rules, alpha=0.3, color="#A23B72")
ax2.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
ax2.set_ylabel("Number of Rules Generated", fontsize=12, fontweight="bold")
ax2.set_title(
    "Confidence Threshold vs Number of Rules", fontsize=13, fontweight="bold", pad=15
)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_xticks(confidences)
ax2.ticklabel_format(style="plain", axis="y")
for i, (conf, rules) in enumerate(zip(confidences, num_rules)):
    ax2.annotate(
        f"{rules:,}",
        (conf, rules),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )

# Graph 3: Dual-axis plot - Confidence vs Time and Rules
ax3 = plt.subplot(2, 2, 3)
ax3_twin = ax3.twinx()

line1 = ax3.plot(
    confidences,
    execution_times,
    marker="o",
    linewidth=2,
    markersize=8,
    color="#2E86AB",
    label="Execution Time",
)
ax3.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
ax3.set_ylabel(
    "Execution Time (seconds)", fontsize=12, fontweight="bold", color="#2E86AB"
)
ax3.tick_params(axis="y", labelcolor="#2E86AB")

line2 = ax3_twin.plot(
    confidences,
    num_rules,
    marker="s",
    linewidth=2,
    markersize=8,
    color="#A23B72",
    label="Number of Rules",
)
ax3_twin.set_ylabel("Number of Rules", fontsize=12, fontweight="bold", color="#A23B72")
ax3_twin.tick_params(axis="y", labelcolor="#A23B72")
ax3_twin.ticklabel_format(style="plain", axis="y")

ax3.set_title(
    "Confidence Threshold: Time vs Rules (Dual Axis)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.set_xticks(confidences)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="upper right", fontsize=10)

# Graph 4: Total Time (A-Priori + Rule Generation)
ax4 = plt.subplot(2, 2, 4)
ax4.plot(
    confidences,
    total_times,
    marker="^",
    linewidth=2,
    markersize=8,
    color="#F18F01",
    label="Total Time",
)
ax4.axhline(
    y=apriori_time,
    color="r",
    linestyle="--",
    linewidth=2,
    label="A-Priori Time (constant)",
)
ax4.fill_between(confidences, total_times, alpha=0.3, color="#F18F01")
ax4.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
ax4.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
ax4.set_title(
    "Total Execution Time\n(A-Priori + Rule Generation)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax4.grid(True, alpha=0.3, linestyle="--")
ax4.set_xticks(confidences)
ax4.legend(fontsize=10)
for i, (conf, total) in enumerate(zip(confidences, total_times)):
    ax4.annotate(
        f"{total:.4f}s",
        (conf, total),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )

# Add overall title
fig.suptitle(
    "Confidence Threshold Analysis: Execution Time and Rule Generation",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save the figure
graph_filename = "hw2/confidence_time_measurement.png"
plt.savefig(graph_filename, dpi=300, bbox_inches="tight")
print(f"  ✓ Graph saved to {graph_filename}")

# Also create a separate detailed graph
fig2, ax = plt.subplots(figsize=(12, 7))

# Plot execution time with more details
ax.plot(
    confidences,
    execution_times,
    marker="o",
    linewidth=3,
    markersize=12,
    color="#2E86AB",
    label="Rule Generation Time",
    zorder=3,
)
ax.scatter(confidences, execution_times, s=200, color="#2E86AB", zorder=4)

# Add value labels
for conf, time in zip(confidences, execution_times):
    ax.annotate(
        f"{time:.4f}s\n({time*1000:.2f}ms)",
        (conf, time),
        textcoords="offset points",
        xytext=(0, 20),
        ha="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    )

ax.set_xlabel("Confidence Threshold", fontsize=14, fontweight="bold")
ax.set_ylabel("Execution Time (seconds)", fontsize=14, fontweight="bold")
ax.set_title(
    "Confidence Threshold vs Execution Time\n(Association Rule Generation)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax.grid(True, alpha=0.4, linestyle="--", linewidth=1)
ax.set_xticks(confidences)
ax.legend(fontsize=12, loc="best")

# Add statistics text box
stats_text = f"Statistics:\n"
stats_text += f"Min Time: {min(execution_times):.4f}s (conf={confidences[execution_times.index(min(execution_times))]:.1f})\n"
stats_text += f"Max Time: {max(execution_times):.4f}s (conf={confidences[execution_times.index(max(execution_times))]:.1f})\n"
stats_text += f"Time Range: {max(execution_times) - min(execution_times):.4f}s\n"
stats_text += f"Speedup: {max(execution_times) / min(execution_times):.2f}x"

ax.text(
    0.02,
    0.98,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

plt.tight_layout()
detailed_graph_filename = "hw2/confidence_time_detailed.png"
plt.savefig(detailed_graph_filename, dpi=300, bbox_inches="tight")
print(f"  ✓ Detailed graph saved to {detailed_graph_filename}")

# Show plots (optional - comment out if running in headless environment)
# plt.show()

print("  ✓ All graphs generated successfully!")

# Cleanup
transactions_rdd.unpersist()
spark.stop()
