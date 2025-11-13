s1 = 1
s2 = 2
s3 = 3
s4 = 4


# This is apriori algorithm data extraction function
# # It reads data from "hw2/data/data.dat" file and returns a list of transactions.
## Each transaction is represented as a list of items.


def extract_data():
    data = []

    with open("hw2/data/data.dat", "r") as file:
        for line in file:
            data.append(line.strip())
    return data


# Create map store item to its support count
def create_single_item_support_map(data):
    item_support_map = {}
    for transaction in data:
        items = transaction.split()
        for item in items:
            if item in item_support_map:
                item_support_map[item] += 1
            else:
                item_support_map[item] = 1
    return item_support_map


def find_frequent_items(item_support_map, min_support):
    frequent_items = {}
    for item, support in item_support_map.items():
        if support >= min_support:
            frequent_items[item] = support
    return frequent_items


# from  the frequent single items, generate candidate itemsets of size 2
def generate_candidate_itemsets(frequent_items, k):
    from itertools import combinations

    items = list(frequent_items.keys())
    candidate_itemsets = list(combinations(items, k))
    return candidate_itemsets


# from the candidate itemsets of size 2, create support map
def create_double_itemset_support_map(data, candidate_itemsets):
    itemset_support_map = {}
    for itemset in candidate_itemsets:
        itemset_support_map[itemset] = 0

    for transaction in data:
        items = set(transaction.split())
        for itemset in candidate_itemsets:
            if all(item in items for item in itemset):
                itemset_support_map[itemset] += 1
    return itemset_support_map


# from the frequent double items, generate candidate itemsets of size 3
# e.g: (A, B), (A, C), (B, C) -> (A, B, C)
#      (A, B) , (A, D) !-> (A, B, D) since (B, D) is not frequent
def generate_candidate_itemsets_size_3(frequent_double_items):
    from itertools import combinations

    items = set()
    for itemset in frequent_double_items.keys():
        items.update(itemset)

    items = sorted(items)
    candidate_itemsets = []

    for combo in combinations(items, 3):
        a, b, c = combo
        if (
            (a, b) in frequent_double_items
            and (a, c) in frequent_double_items
            and (b, c) in frequent_double_items
        ):
            candidate_itemsets.append(combo)
    return candidate_itemsets


# filter out items below min support threshold
min_support_threshold = 1000

data = extract_data()

item_support_map = create_single_item_support_map(data)

frequent_single_items = find_frequent_items(item_support_map, min_support_threshold)
print("Frequent single Items:", frequent_single_items)


candidate_itemsets_size_2 = generate_candidate_itemsets(frequent_single_items, 2)
# print("Candidate Itemsets of size 2:", candidate_itemsets_size_2)


double_itemset_support_map = create_double_itemset_support_map(
    data, candidate_itemsets_size_2
)
# print("Double Itemset Support Map:", double_itemset_support_map)

frequent_double_items = find_frequent_items(
    double_itemset_support_map, min_support_threshold
)
print("Frequent double Items:", frequent_double_items)

triple_itemset_candidates = generate_candidate_itemsets_size_3(frequent_double_items)
# print("Candidate Itemsets of size 3:", triple_itemset_candidates)

triple_itemset_support_map = create_double_itemset_support_map(
    data, triple_itemset_candidates
)
# print("Triple Itemset Support Map:", triple_itemset_support_map)
frequent_triple_items = find_frequent_items(
    triple_itemset_support_map, min_support_threshold
)
print("Frequent triple Items:", frequent_triple_items)
