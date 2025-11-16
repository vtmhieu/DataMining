from pyspark.sql import SparkSession
from pyspark import StorageLevel
from itertools import combinations
import time

# Initialize Spark with optimized configuration
import multiprocessing

num_cores = multiprocessing.cpu_count()

spark = (
    SparkSession.builder.appName("HighPerformanceApriori")
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
min_confidence_threshold = 0.6

print("=" * 70)
print("HIGH-PERFORMANCE APRIORI WITH SPARK - ASSOCIATION RULES")
print("=" * 70)
print(
    f"Min Support: {min_support_threshold} | Min Confidence: {min_confidence_threshold}"
)
print("=" * 70)

start_total = time.time()

# ============================================================================
# STEP 1: LOAD AND OPTIMIZE DATA
# ============================================================================
print("\n[1/5] Loading and preprocessing data...")
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
# STEP 2: FIND FREQUENT SINGLE ITEMS (L1)
# ============================================================================
print("\n[2/5] Mining frequent single items...")
start = time.time()

# Count single items
single_item_counts = (
    transactions_rdd.flatMap(lambda txn: txn)
    .map(lambda item: (item, 1))
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support_threshold)
)

frequent_single_dict = dict(single_item_counts.collect())
print(
    f"  ✓ Found {len(frequent_single_dict):,} frequent items in {time.time()-start:.2f}s"
)

# ============================================================================
# STEP 3: FIND FREQUENT PAIRS (L2)
# ============================================================================
print("\n[3/5] Mining frequent pairs...")
start = time.time()

# Broadcast frequent items
frequent_items_set = set(frequent_single_dict.keys())
frequent_items_bc = sc.broadcast(frequent_items_set)


def generate_pairs(txn):
    """Generate pairs from transaction"""
    # Filter to frequent items and sort
    items = sorted([item for item in txn if item in frequent_items_bc.value])

    if len(items) < 2:
        return []

    # Generate all pairs
    return [
        ((items[i], items[j]), 1)
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


# Count pairs
double_itemset_counts = (
    transactions_rdd.flatMap(generate_pairs)
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support_threshold)
)

frequent_double_dict = dict(double_itemset_counts.collect())
print(
    f"  ✓ Found {len(frequent_double_dict):,} frequent pairs in {time.time()-start:.2f}s"
)

# ============================================================================
# STEP 4: FIND FREQUENT TRIPLES (L3)
# ============================================================================
print("\n[4/5] Mining frequent triples...")
start = time.time()


# Generate candidate triples with Apriori pruning
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
print(f"  - Generated {len(candidate_triples):,} candidate triples")

# Broadcast candidates
candidate_triples_bc = sc.broadcast(set(candidate_triples))


def generate_triples(txn):
    """Generate triples from transaction"""
    # Sort transaction items
    items = sorted(txn)

    if len(items) < 3:
        return []

    # Generate all triples and filter by candidates
    candidates_set = candidate_triples_bc.value
    triples = []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for k in range(j + 1, len(items)):
                triple = (items[i], items[j], items[k])
                if triple in candidates_set:
                    triples.append((triple, 1))

    return triples


# Count triples
triple_itemset_counts = (
    transactions_rdd.flatMap(generate_triples)
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support_threshold)
)

frequent_triple_dict = dict(triple_itemset_counts.collect())
print(
    f"  ✓ Found {len(frequent_triple_dict):,} frequent triples in {time.time()-start:.2f}s"
)

# ============================================================================
# STEP 5: GENERATE ASSOCIATION RULES
# ============================================================================
print("\n[5/5] Generating association rules...")
start = time.time()

# Create unified support map for lookups
unified_support_map = {}

# Add single items as tuples
for item, support in frequent_single_dict.items():
    unified_support_map[(item,)] = support

# Add pairs and triples
unified_support_map.update(frequent_double_dict)
unified_support_map.update(frequent_triple_dict)

# Broadcast support map for distributed processing
support_map_bc = sc.broadcast(unified_support_map)


def generate_rules_from_itemset(itemset_and_support):
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

            if confidence >= min_confidence_threshold:
                rules.append((antecedent, consequent, confidence, support, ant_support))

    return rules


# Combine pairs and triples for rule generation
all_frequent_itemsets = list(frequent_double_dict.items()) + list(
    frequent_triple_dict.items()
)

# Generate rules in parallel
if all_frequent_itemsets:
    rules_rdd = sc.parallelize(all_frequent_itemsets)
    all_rules = rules_rdd.flatMap(generate_rules_from_itemset).collect()

    # Sort by confidence
    all_rules.sort(key=lambda x: x[2], reverse=True)
else:
    all_rules = []

print(f"  ✓ Generated {len(all_rules):,} association rules in {time.time()-start:.2f}s")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total execution time: {time.time() - start_total:.2f}s")
print(f"\nFrequent Itemsets:")
print(f"  L1 (singles):  {len(frequent_single_dict):,}")
print(f"  L2 (pairs):    {len(frequent_double_dict):,}")
print(f"  L3 (triples):  {len(frequent_triple_dict):,}")
print(f"\nAssociation Rules: {len(all_rules):,}")

# Display top single items
if frequent_single_dict:
    print(f"\nTop 10 Frequent Single Items:")
    top_singles = sorted(
        frequent_single_dict.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for item, count in top_singles:
        print(f"  {item:>6s}: {count:>6,}")

# Display top pairs
if frequent_double_dict:
    print(f"\nTop 10 Frequent Pairs:")
    top_pairs = sorted(frequent_double_dict.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    for itemset, count in top_pairs:
        print(f"  {str(itemset):>20s}: {count:>6,}")

# Display top triples
if frequent_triple_dict:
    print(f"\nTop 10 Frequent Triples:")
    top_triples = sorted(
        frequent_triple_dict.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for itemset, count in top_triples:
        print(f"  {str(itemset):>30s}: {count:>6,}")

# Display association rules
if all_rules:
    print(f"\nTop 20 Association Rules (by confidence):")
    print(f"{'Antecedent':<30} {'Consequent':<30} {'Confidence':>10}")
    print("-" * 72)

    for antecedent, consequent, confidence, support, ant_support in all_rules[:20]:
        ant_str = str(antecedent)
        cons_str = str(consequent)
        print(f"{ant_str:<30} => {cons_str:<30} {confidence:>10.4f}")

    if len(all_rules) > 20:
        print(f"\n... and {len(all_rules) - 20} more rules")

print("\n" + "=" * 70)

# Also print results in original format
print("\nFrequent single Items:", frequent_single_dict)
print("\nFrequent double Items:", frequent_double_dict)
print("\nFrequent triple Items:", frequent_triple_dict)

print("\nAssociation Rules:")
for antecedent, consequent, confidence, support, ant_support in all_rules:
    print(f"{antecedent} -> {consequent} with confidence {confidence:.2f}")

# Cleanup
transactions_rdd.unpersist()
spark.stop()
