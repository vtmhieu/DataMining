from pyspark import SparkContext, SparkConf
from itertools import combinations
import time

# Initialize Spark
conf = SparkConf().setAppName("GeneralizedApriori").setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

# Configuration
min_support_threshold = 3
max_k = None  # Set to a number to limit max itemset size, None for unlimited

print("=" * 70)
print("GENERALIZED A-PRIORI ALGORITHM (ANY K)")
print("=" * 70)
print(f"Min Support: {min_support_threshold}")
print(f"Max K: {max_k if max_k else 'unlimited'}")
print("=" * 70)

start_total = time.time()

# Load and parse data
print("\nLoading data...")
start = time.time()
data_rdd = sc.textFile("hw2/data/data2.dat")
transactions_rdd = data_rdd.map(lambda line: line.strip().split()).cache()
num_transactions = transactions_rdd.count()
print(f"  Loaded {num_transactions:,} transactions in {time.time()-start:.2f}s")


# ============================================================================
# STEP 1: Find frequent single items (L1)
# ============================================================================
def find_frequent_single_items(transactions_rdd, min_support):
    """Find frequent single items (L1)"""
    item_counts = (
        transactions_rdd.flatMap(lambda transaction: transaction)
        .map(lambda item: (item, 1))
        .reduceByKey(lambda a, b: a + b)
        .filter(lambda x: x[1] >= min_support)
    )
    return dict(item_counts.collect())


print("\n[K=1] Finding frequent single items...")
start = time.time()
frequent_single_dict = find_frequent_single_items(
    transactions_rdd, min_support_threshold
)
print(
    f"  Found {len(frequent_single_dict):,} frequent items in {time.time()-start:.2f}s"
)

if not frequent_single_dict:
    print("No frequent items found. Stopping.")
    sc.stop()
    exit()

# Store all frequent itemsets
all_frequent_itemsets = {1: frequent_single_dict}


# ============================================================================
# GENERALIZED FUNCTIONS FOR ANY K
# ============================================================================
def generate_candidates_k(frequent_k_minus_1_dict, k):
    """
    Generate candidate itemsets of size k from frequent itemsets of size k-1.
    Uses A-Priori pruning: only generate candidates where all (k-1)-subsets are frequent.

    Args:
        frequent_k_minus_1_dict: Dictionary of frequent (k-1)-itemsets
        k: Size of candidates to generate

    Returns:
        List of candidate k-itemsets (tuples)
    """
    if k < 2:
        return []

    # For k=2, generate from single items
    if k == 2:
        items = sorted(frequent_k_minus_1_dict.keys())
        return list(combinations(items, 2))

    # For k>=3, use A-Priori pruning
    # Collect all items from frequent (k-1)-itemsets
    items = set()
    for itemset in frequent_k_minus_1_dict.keys():
        items.update(itemset)
    items = sorted(items)

    if len(items) < k:
        return []

    # Convert to set for O(1) lookup
    frequent_set = set(frequent_k_minus_1_dict.keys())
    candidates = []

    # Generate all k-combinations and check if all (k-1)-subsets are frequent
    for combo in combinations(items, k):
        # Check all (k-1)-subsets
        all_subsets_frequent = True
        for i in range(k):
            # Create subset by removing element at index i
            subset = combo[:i] + combo[i + 1 :]
            if subset not in frequent_set:
                all_subsets_frequent = False
                break

        if all_subsets_frequent:
            candidates.append(combo)

    return candidates


def count_itemset_support_k(transactions_rdd, candidates, k, sc):
    """
    Count support for candidate k-itemsets.

    Args:
        transactions_rdd: RDD of transactions
        candidates: List of candidate itemsets (tuples)
        k: Size of itemsets
        sc: SparkContext

    Returns:
        Dictionary of itemset -> support count
    """
    if not candidates:
        return {}

    # Broadcast candidates for efficient lookup
    candidates_broadcast = sc.broadcast(set(candidates))

    def map_transaction(transaction):
        """Generate k-itemsets from transaction and filter by candidates"""
        transaction_items = sorted(transaction)

        if len(transaction_items) < k:
            return []

        # Generate all k-combinations from this transaction
        transaction_candidates = list(combinations(transaction_items, k))

        # Filter to only keep candidates that are in our candidate set
        candidate_set = candidates_broadcast.value
        results = []
        for candidate in transaction_candidates:
            if candidate in candidate_set:
                results.append((candidate, 1))
        return results

    itemset_counts = transactions_rdd.flatMap(map_transaction).reduceByKey(
        lambda a, b: a + b
    )

    return dict(itemset_counts.collect())


def find_frequent_itemsets_k(support_dict, min_support):
    """Filter itemsets by minimum support"""
    return {
        itemset: support
        for itemset, support in support_dict.items()
        if support >= min_support
    }


# ============================================================================
# ITERATIVE MINING FOR K >= 2
# ============================================================================
k = 2
frequent_k_minus_1 = frequent_single_dict

while frequent_k_minus_1 and (max_k is None or k <= max_k):
    print(f"\n[K={k}] Finding frequent {k}-itemsets...")
    start = time.time()

    # Step 1: Generate candidates
    candidates = generate_candidates_k(frequent_k_minus_1, k)

    if not candidates:
        print(f"  No candidates generated (A-Priori pruning). Stopping.")
        break

    print(f"  Generated {len(candidates):,} candidates")

    # Step 2: Count support
    support_dict = count_itemset_support_k(transactions_rdd, candidates, k, sc)

    # Step 3: Filter by minimum support
    frequent_k = find_frequent_itemsets_k(support_dict, min_support_threshold)

    elapsed = time.time() - start

    if not frequent_k:
        print(f"  No frequent {k}-itemsets found. Stopping.")
        break

    print(f"  Found {len(frequent_k):,} frequent {k}-itemsets in {elapsed:.2f}s")

    # Store results
    all_frequent_itemsets[k] = frequent_k

    # Prepare for next iteration
    frequent_k_minus_1 = frequent_k
    k += 1


# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total execution time: {time.time() - start_total:.2f}s")
print(f"Minimum support: {min_support_threshold:,}")

total_itemsets = sum(len(itemsets) for itemsets in all_frequent_itemsets.values())
print(f"\nTotal frequent itemsets found: {total_itemsets:,}")

print("\nFrequent itemsets by size:")
for k in sorted(all_frequent_itemsets.keys()):
    print(f"  L{k}: {len(all_frequent_itemsets[k]):,} itemsets")

# Display detailed results for each k
for k in sorted(all_frequent_itemsets.keys()):
    frequent_k = all_frequent_itemsets[k]
    print(f"\n{'='*70}")
    print(f"Frequent {k}-itemsets (L{k}): {len(frequent_k):,} itemsets")
    print(f"{'='*70}")

    # Sort by support and show top 20
    sorted_itemsets = sorted(frequent_k.items(), key=lambda x: x[1], reverse=True)
    display_count = min(20, len(sorted_itemsets))

    for i, (itemset, support) in enumerate(sorted_itemsets[:display_count], 1):
        # Format itemset nicely
        if k == 1:
            itemset_str = f"('{itemset}',)"
        else:
            itemset_str = str(itemset)

        support_pct = support / num_transactions * 100
        print(
            f"{i:2d}. {itemset_str:<50s} support={support:>6,} ({support_pct:>5.2f}%)"
        )

    if len(sorted_itemsets) > display_count:
        print(f"\n... and {len(sorted_itemsets) - display_count:,} more itemsets")

print("\n" + "=" * 70)

# Print in original format for compatibility
print("\nOriginal Format Output:")
print("Frequent single Items:", all_frequent_itemsets.get(1, {}))
if 2 in all_frequent_itemsets:
    print("Frequent double Items:", all_frequent_itemsets[2])
if 3 in all_frequent_itemsets:
    print("Frequent triple Items:", all_frequent_itemsets[3])

# Stop Spark context
sc.stop()
