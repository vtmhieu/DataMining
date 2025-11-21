import random
import sys
import time
from collections import defaultdict

# ==========================================
# PART 1: CORE RESERVOIR SAMPLING
# ==========================================


def sample_edge(t, M, sample_set):
    """
    Decides whether to insert the t-th edge into the sample of size M.
    Returns: (should_insert, edge_to_remove)
    """
    if t <= M:
        # Phase 1: Fill the reservoir
        return True, None
    else:
        # Phase 2: Probabilistic replacement
        # Probability of keeping the new item is M / t [cite: 98-101]
        if random.random() < (M / t):
            # Select a uniform random edge to remove from S
            removed_edge = random.choice(list(sample_set))
            return True, removed_edge
        else:
            return False, None


# ==========================================
# PART 2: TRIÈST-BASE IMPLEMENTATION
# ==========================================


class TriestBase:
    def __init__(self, M):
        self.M = M
        self.S = set()  # Reservoir of edges
        self.adj = defaultdict(set)  # Adjacency list of S
        self.t = 0  # Total edges processed
        self.global_tau = 0  # Global triangle counter
        self.local_tau = defaultdict(int)  # Local triangle counters

    def get_shared_neighbors(self, u, v):
        # Return intersection of neighbors in Sample S
        if u not in self.adj or v not in self.adj:
            return set()
        return self.adj[u].intersection(self.adj[v])

    def update_counters(self, operation, neighbors, u, v):
        val = 1 if operation == "+" else -1
        count = len(neighbors)
        if count == 0:
            return

        # Update Global
        self.global_tau += val * count

        # Update Local
        self.local_tau[u] += val * count
        self.local_tau[v] += val * count
        for c in neighbors:
            self.local_tau[c] += val

    def process_edge(self, u, v):
        self.t += 1
        edge = tuple(sorted((u, v)))

        # Reservoir Sampling Logic
        insert, removed = sample_edge(self.t, self.M, self.S)

        if insert:
            # 1. If we are removing an edge, update counters NEGATIVELY first
            if removed:
                ru, rv = removed
                shared_rem = self.get_shared_neighbors(ru, rv)
                self.update_counters("-", shared_rem, ru, rv)

                # Remove from S and Adj
                self.S.remove(removed)
                self.adj[ru].remove(rv)
                self.adj[rv].remove(ru)
                if not self.adj[ru]:
                    del self.adj[ru]
                if not self.adj[rv]:
                    del self.adj[rv]

            # 2. Insert new edge and update counters POSITIVELY
            shared_new = self.get_shared_neighbors(u, v)
            self.update_counters("+", shared_new, u, v)

            # Add to S and Adj
            self.S.add(edge)
            self.adj[u].add(v)
            self.adj[v].add(u)

    def get_estimation(self):
        # Estimation = tau * xi(t) [cite: 105-107]
        # xi(t) = max(1, (t*(t-1)*(t-2)) / (M*(M-1)*(M-2)))
        if self.t < 3:
            return 0

        num = self.t * (self.t - 1) * (self.t - 2)
        den = self.M * (self.M - 1) * (self.M - 2)
        xi = max(1.0, num / den)

        return int(self.global_tau * xi)


# ==========================================
# PART 3: TRIÈST-IMPR IMPLEMENTATION
# ==========================================


class TriestImpr:
    def __init__(self, M):
        self.M = M
        self.S = set()
        self.adj = defaultdict(set)
        self.t = 0
        self.global_est = 0.0  # Kept as float for weights
        self.local_est = defaultdict(float)

    def get_shared_neighbors(self, u, v):
        if u not in self.adj or v not in self.adj:
            return set()
        return self.adj[u].intersection(self.adj[v])

    def get_weight(self):
        # Weight eta(t) [cite: 164]
        # Update happens BEFORE t is incremented in algorithm logic, so we use t-1 logic
        t_curr = self.t
        if t_curr <= self.M:
            return 1.0
        else:
            num = (t_curr - 1) * (t_curr - 2)
            den = self.M * (self.M - 1)
            return max(1.0, num / den)

    def process_edge(self, u, v):
        self.t += 1
        edge = tuple(sorted((u, v)))

        # 1. Update counters UNCONDITIONALLY before sampling [cite: 158]
        shared = self.get_shared_neighbors(u, v)
        if shared:
            w = self.get_weight()
            count = len(shared)
            self.global_est += w * count
            self.local_est[u] += w * count
            self.local_est[v] += w * count
            for c in shared:
                self.local_est[c] += w

        # 2. Reservoir Logic
        insert, removed = sample_edge(self.t, self.M, self.S)

        if insert:
            if removed:
                ru, rv = removed
                # Remove from S/Adj ONLY (No counter decrement) [cite: 162]
                self.S.remove(removed)
                self.adj[ru].remove(rv)
                self.adj[rv].remove(ru)
                if not self.adj[ru]:
                    del self.adj[ru]
                if not self.adj[rv]:
                    del self.adj[rv]

            # Insert into S/Adj
            self.S.add(edge)
            self.adj[u].add(v)
            self.adj[v].add(u)

    def get_estimation(self):
        # For IMPR, the counter IS the estimate [cite: 168]
        return int(self.global_est)


# ==========================================
# HELPER: EXACT COUNTER (GROUND TRUTH)
# ==========================================


class ExactCounter:
    """Naive in-memory exact counter for verification"""

    def __init__(self):
        self.adj = defaultdict(set)
        self.count = 0

    def process_edge(self, u, v):
        # Check intersection
        if u in self.adj and v in self.adj:
            shared = self.adj[u].intersection(self.adj[v])
            self.count += len(shared)

        # Add edge
        self.adj[u].add(v)
        self.adj[v].add(u)


# ==========================================
# PART 4: EXPERIMENT RUNNER
# ==========================================


def read_dataset(filepath):
    """Generator that yields edges (u, v) from a file"""
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("%"):
                continue
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:  # Ignore self-loops
                        yield u, v
                except ValueError:
                    continue


def main():
    # --- CONFIGURATION ---
    # Replace this with your downloaded dataset filename
    # Dataset example: http://snap.stanford.edu/data/facebook_combined.txt.gz
    dataset_path = "hw3/web-Google.txt"
    M = 5000  # Memory size (Sample size)

    # Check if file exists
    import os

    if not os.path.exists(dataset_path):
        print(f"Error: File '{dataset_path}' not found.")
        print(
            "Please download a dataset (e.g., from SNAP 'Web graphs') and place it in this folder."
        )
        return

    print(f"--- Processing {dataset_path} with M={M} ---")

    # Initialize Algorithms
    base_algo = TriestBase(M)
    impr_algo = TriestImpr(M)
    exact_algo = ExactCounter()

    start_time = time.time()

    # Stream Processing Loop
    edge_count = 0
    for u, v in read_dataset(dataset_path):
        edge_count += 1

        # Run Algorithms
        base_algo.process_edge(u, v)
        impr_algo.process_edge(u, v)
        exact_algo.process_edge(u, v)

        if edge_count % 10000 == 0:
            sys.stdout.write(f"\rProcessed {edge_count} edges...")
            sys.stdout.flush()

    print(f"\n\nStream ended. Total edges (T): {edge_count}")
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    # --- RESULTS & REPORTING ---

    true_count = exact_algo.count
    base_est = base_algo.get_estimation()
    impr_est = impr_algo.get_estimation()

    def calculate_mape(est, true_val):
        if true_val == 0:
            return 0.0
        return abs(true_val - est) / true_val

    print("-" * 50)
    print("FINAL RESULTS")
    print("-" * 50)
    print(f"{'Algorithm':<15} | {'Triangle Count':<15} | {'Error (MAPE)':<10}")
    print("-" * 50)
    print(f"{'Ground Truth':<15} | {true_count:<15} | {'0.0%':<10}")
    print(
        f"{'TRIEST-BASE':<15} | {base_est:<15} | {calculate_mape(base_est, true_count):.2%}"
    )
    print(
        f"{'TRIEST-IMPR':<15} | {impr_est:<15} | {calculate_mape(impr_est, true_count):.2%}"
    )
    print("-" * 50)

    print("\n[Analysis Notes for Report]")
    if abs(impr_est - true_count) < abs(base_est - true_count):
        print("Observation: TRIEST-IMPR provided a more accurate estimate than BASE.")
    print(
        f"Memory Usage: The algorithm only stored ~{len(base_algo.S)} edges in memory."
    )


if __name__ == "__main__":
    main()
