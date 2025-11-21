import random
import math


def SAMPLE_EDGE(edge, current_time, M, sample_set):
    """
    Implements Reservoir Sampling for an incoming edge (u, v).

    Args:
        edge (tuple): The edge (u, v) from the stream.
        current_time (int): The total number of edges seen so far (t).
        M (int): The maximum size of the sample (M).
        sample_set (set): The current set of sampled edges (S).

    Returns:
        tuple: (boolean, removed_edge). True if the edge is inserted into S.
               removed_edge is the edge to be removed, or None.
    """
    removed_edge = None

    # Rule 1: t <= M (initial deterministic insertion) [cite: 99]
    if current_time <= M:
        return True, removed_edge

    # Rule 2: t > M (probabilistic replacement) [cite: 100]
    else:
        # Flip a biased coin with probability M/t
        probability = M / current_time
        if random.random() < probability:
            # Choose an edge (w, z) from S uniformly at random [cite: 101]
            removed_edge = random.choice(list(sample_set))
            # The calling function (TRIEST_BASE) handles removal/insertion logic
            return True, removed_edge
        else:
            # S is not modified [cite: 101]
            return False, removed_edge
