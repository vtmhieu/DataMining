"""
Association Rules Generation - General Implementation with Spark RDD
Generates association rules from frequent itemsets discovered by A-Priori algorithm.

An association rule is an implication X -> Y, where X and Y are itemsets such that
X ∩ Y = ∅. Support of rule X -> Y is the number of transactions containing X ∪ Y.
Confidence of rule X -> Y is support(X ∪ Y) / support(X).
"""

import argparse
from pyspark import SparkContext, SparkConf
from itertools import combinations


def initialize_spark(app_name="AssociationRules"):
    """
    Initialize Spark context with optimized configuration.

    Args:
        app_name: Name of the Spark application

    Returns:
        SparkContext
    """
    conf = SparkConf().setAppName(app_name).setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc


def load_frequent_itemsets_from_apriori(all_frequent_itemsets):
    """
    Convert the output from apriori_algorithm to a unified support map.

    Args:
        all_frequent_itemsets: Dictionary mapping k -> frequent k-itemsets (itemset -> support)

    Returns:
        Dictionary mapping itemset (tuple) -> support count
    """
    unified_support_map = {}

    for k, frequent_k in all_frequent_itemsets.items():
        for itemset, support in frequent_k.items():
            # Convert single items to tuples for consistency
            if k == 1:
                itemset_tuple = (itemset,) if isinstance(itemset, str) else itemset
            else:
                itemset_tuple = itemset if isinstance(itemset, tuple) else tuple(itemset)

            unified_support_map[itemset_tuple] = support

    return unified_support_map


def generate_rules_from_itemset(itemset_and_support, support_map_bc, min_confidence):
    """
    Generate all valid association rules from a single frequent itemset.

    For itemset I with support s, generate all rules X -> Y where:
    - X ∪ Y = I, X ∩ Y = ∅
    - confidence = s / support(X) >= min_confidence

    Args:
        itemset_and_support: Tuple of (itemset, support_count)
        support_map_bc: Broadcast variable containing support map
        min_confidence: Minimum confidence threshold

    Returns:
        List of rules, each as (antecedent, consequent, confidence, support, antecedent_support)
    """
    itemset, itemset_support = itemset_and_support
    rules = []

    items = list(itemset)
    n = len(items)

    if n < 2:
        return rules  # Need at least 2 items to form a rule

    support_map = support_map_bc.value

    # Generate all possible splits: X -> Y where X ∪ Y = itemset
    for i in range(1, n):
        # Generate all possible antecedents of size i
        for antecedent_items in combinations(items, i):
            antecedent = tuple(sorted(antecedent_items))
            consequent_items = tuple(sorted(set(items) - set(antecedent)))

            if len(consequent_items) == 0:
                continue

            # Get support of antecedent
            antecedent_support = support_map.get(antecedent, 0)

            if antecedent_support == 0:
                continue  # Antecedent is not frequent, skip

            # Calculate confidence
            confidence = itemset_support / antecedent_support

            if confidence >= min_confidence:
                rules.append(
                    (
                        antecedent,
                        consequent_items,
                        confidence,
                        itemset_support,
                        antecedent_support,
                    )
                )

    return rules


def generate_association_rules_rdd(sc, all_frequent_itemsets, min_confidence):
    """
    Generate all association rules from frequent itemsets using Spark RDD.

    Args:
        sc: SparkContext
        all_frequent_itemsets: Dictionary mapping k -> frequent k-itemsets (itemset -> support)
        min_confidence: Minimum confidence threshold

    Returns:
        List of rules, each as (antecedent, consequent, confidence, support, antecedent_support)
    """
    # Create unified support map
    support_map = load_frequent_itemsets_from_apriori(all_frequent_itemsets)

    # Broadcast support map for distributed processing
    support_map_bc = sc.broadcast(support_map)

    # Collect all frequent itemsets of size >= 2 for rule generation
    all_itemsets_for_rules = []
    for k in sorted(all_frequent_itemsets.keys()):
        if k < 2:
            continue  # Need at least 2 items to form a rule

        frequent_k = all_frequent_itemsets[k]
        for itemset, support in frequent_k.items():
            # Ensure itemset is a tuple
            if isinstance(itemset, str):
                itemset_tuple = (itemset,)
            elif isinstance(itemset, tuple):
                itemset_tuple = itemset
            else:
                itemset_tuple = tuple(itemset)

            all_itemsets_for_rules.append((itemset_tuple, support))

    if not all_itemsets_for_rules:
        return []

    # Create RDD from itemsets and generate rules in parallel
    itemsets_rdd = sc.parallelize(all_itemsets_for_rules)

    # Generate rules using flatMap
    rules_rdd = itemsets_rdd.flatMap(
        lambda x: generate_rules_from_itemset(x, support_map_bc, min_confidence)
    )

    # Collect all rules
    all_rules = rules_rdd.collect()

    # Sort by confidence (descending)
    all_rules.sort(key=lambda x: x[2], reverse=True)

    return all_rules


def print_rules(all_rules, top_n=20):
    """
    Print association rules in a readable format.

    Args:
        all_rules: List of rules
        top_n: Number of top rules to display (sorted by confidence)
    """
    print("\n" + "=" * 70)
    print("ASSOCIATION RULES RESULTS")
    print("=" * 70)

    if not all_rules:
        print("No association rules found.")
        return

    print(f"\nTotal rules found: {len(all_rules)}")
    print(f"\nTop {min(top_n, len(all_rules))} rules (by confidence):")
    print(
        f"{'Antecedent':<30} {'Consequent':<30} {'Confidence':>12} {'Support':>10}"
    )
    print("-" * 84)

    for (
        antecedent,
        consequent,
        confidence,
        support,
        ant_support,
    ) in all_rules[:top_n]:
        ant_str = str(antecedent)
        cons_str = str(consequent)
        print(f"{ant_str:<30} => {cons_str:<30} {confidence:>12.4f} {support:>10}")

    if len(all_rules) > top_n:
        print(f"\n... and {len(all_rules) - top_n} more rules")

    print("\n" + "=" * 70)

    # Also print in original format for compatibility
    print("\nAll Association Rules:")
    for (
        antecedent,
        consequent,
        confidence,
        support,
        ant_support,
    ) in all_rules:
        print(
            f"{antecedent} -> {consequent} with confidence {confidence:.4f} (support: {support}, antecedent support: {ant_support})"
        )


def main():
    """
    Main function to generate association rules with Spark RDD.
    This function can be used standalone or imported by other modules.
    """
    parser = argparse.ArgumentParser(
        description="Generate association rules from frequent itemsets (Spark RDD)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="hw2/data/data.dat",
        help="Path to data file (default: hw2/data/data.dat)",
    )
    parser.add_argument(
        "--support",
        type=int,
        default=1000,
        help="Minimum support threshold (default: 1000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top rules to display (default: 20)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ASSOCIATION RULES GENERATION - GENERAL IMPLEMENTATION (SPARK RDD)")
    print("=" * 70)
    print(f"Data file: {args.data}")
    print(f"Minimum support: {args.support}")
    print(f"Minimum confidence: {args.confidence}")
    print("=" * 70)

    # Initialize Spark
    sc = initialize_spark("AssociationRules")

    try:
        # Import and run A-Priori algorithm
        try:
            from a_priori_final import (
                load_transactions,
                apriori_algorithm_rdd,
            )

            print("\n[Step 1] Running A-Priori algorithm to find frequent itemsets...")
            transactions_rdd = load_transactions(sc, args.data)
            num_transactions = transactions_rdd.count()
            print(f"Loaded {num_transactions} transactions")

            all_frequent_itemsets = apriori_algorithm_rdd(
                sc, transactions_rdd, args.support
            )

            print("\n[Step 2] Generating association rules...")
            all_rules = generate_association_rules_rdd(
                sc, all_frequent_itemsets, args.confidence
            )

            print_rules(all_rules, args.top)

            # Cleanup
            transactions_rdd.unpersist()

            return all_rules

        except ImportError:
            print("Error: Could not import a_priori_final module.")
            print("Please ensure a_priori_final.py is in the same directory.")
            return None

    finally:
        # Stop Spark context
        sc.stop()


if __name__ == "__main__":
    all_rules = main()
