from pyspark import SparkContext, SparkConf
from itertools import combinations

# Initialize Spark
conf = SparkConf().setAppName("AprioriAlgorithm").setMaster("local[*]")
sc = SparkContext(conf=conf)

# Minimum support threshold
min_support_threshold = 1000

# Load and parse data
data_rdd = sc.textFile("hw2/data/data.dat")
transactions_rdd = data_rdd.map(lambda line: line.strip().split())

# Cache transactions for reuse
transactions_rdd.cache()


# Step 1: Find frequent single items
def create_single_item_support_map(transactions_rdd):
    # Flat map to get all items, then count
    item_counts = (
        transactions_rdd.flatMap(lambda transaction: transaction)
        .map(lambda item: (item, 1))
        .reduceByKey(lambda a, b: a + b)
    )
    return item_counts


def find_frequent_items(item_counts_rdd, min_support):
    # Filter items with support >= min_support
    frequent = item_counts_rdd.filter(lambda x: x[1] >= min_support)
    return frequent


# Get frequent single items
single_item_counts = create_single_item_support_map(transactions_rdd)
frequent_single_items = find_frequent_items(single_item_counts, min_support_threshold)

# Collect to driver for printing
frequent_single_dict = dict(frequent_single_items.collect())
print("Frequent single Items:", frequent_single_dict)


# Step 2: Generate and count candidate itemsets of size 2
def generate_candidate_itemsets(frequent_items_list, k):
    items = sorted(frequent_items_list)
    candidates = list(combinations(items, k))
    return candidates


# Broadcast frequent items for efficiency
frequent_items_list = list(frequent_single_dict.keys())
candidate_itemsets_size_2 = generate_candidate_itemsets(frequent_items_list, 2)
candidate_itemsets_broadcast = sc.broadcast(set(candidate_itemsets_size_2))


def count_itemset_support(transactions_rdd, candidates_broadcast, k):
    def map_transaction(transaction):
        # Generate all k-sized combinations from this transaction
        transaction_items = sorted(transaction)
        if len(transaction_items) < k:
            return []

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
    return itemset_counts


# Count support for size-2 itemsets
double_itemset_counts = count_itemset_support(
    transactions_rdd, candidate_itemsets_broadcast, 2
)
frequent_double_items = find_frequent_items(
    double_itemset_counts, min_support_threshold
)

# Collect to driver
frequent_double_dict = dict(frequent_double_items.collect())
print("Frequent double Items:", frequent_double_dict)


# Step 3: Generate candidate itemsets of size 3
def generate_candidate_itemsets_size_3(frequent_double_dict):
    items = set()
    for itemset in frequent_double_dict.keys():
        items.update(itemset)
    items = sorted(items)

    candidate_itemsets = []
    for combo in combinations(items, 3):
        a, b, c = combo
        if (
            (a, b) in frequent_double_dict
            and (a, c) in frequent_double_dict
            and (b, c) in frequent_double_dict
        ):
            candidate_itemsets.append(combo)
    return candidate_itemsets


# Generate size-3 candidates
triple_itemset_candidates = generate_candidate_itemsets_size_3(frequent_double_dict)
triple_candidates_broadcast = sc.broadcast(set(triple_itemset_candidates))

# Count support for size-3 itemsets
triple_itemset_counts = count_itemset_support(
    transactions_rdd, triple_candidates_broadcast, 3
)
frequent_triple_items = find_frequent_items(
    triple_itemset_counts, min_support_threshold
)

# Collect and print results
frequent_triple_dict = dict(frequent_triple_items.collect())
print("Frequent triple Items:", frequent_triple_dict)

# Stop Spark context
sc.stop()
