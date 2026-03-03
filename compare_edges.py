# compare_edges.py
# Use to compare the edges between 2 different ACDC runs (edges are stored in pickle dump files)
# Use ``python inspect_pkls.py \runs\[The run you intend to see]\another_final_edges.pkl``  to print the edges saved from an ACDC run of your choice
# If the run has not been saved yet (most recent run), you may run ``python inspect_pkls.py`` to inspect it with less hassle

# edges.pkl is expected to be a list of items like: ((dst_hook, dst_index, src_hook, src_index), score) [Essentially, edge ID (Node1, Node2) and its importance score]


import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

EdgeKey = Tuple[Any, Any, Any, Any]   # ('dst', slice, 'src', slice) with weird slice objects
Edge = Tuple[EdgeKey, float]

# Load and validate the edges in edges.pkl
def load_edges(path: Path) -> List[Edge]:
    skipped = 0
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"{path} expected list, got {type(obj)}")
    
    # Ensure it's list[(key, score)]
    out: List[Edge] = []
    for item in obj:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(f"{path} expected tuple(key, score), got {item!r}")
        key, score = item
        
        # Some runs may contain placeholder edges with score 'None'
        if score is None:                       
            skipped += 1
            continue
        out.append((key, float(score)))         # Convert score to a float for numeric comparison
    
    if skipped:
        print(f"{path}: {skipped} edges were skipped due to having an importance score of 'None' recorded")
    
    return out

# Type Conversion to make edge keys hashable
def key_to_str(k: EdgeKey) -> str:
    # repr keeps the "[:]" objects readable
    return repr(k)

# Convert List[(key, score)] to Dict[str_key, score]
def as_dict(edges: Iterable[Edge]) -> Dict[str, float]:
    # canonicalise edge key to string so it can be hashed robustly
    d: Dict[str, float] = {}
    for k, s in edges:
        d[key_to_str(k)] = s
    return d

# Compute the Jaccard similarity of the two sets
# J(A,B) = |A n B| / |A u B|                     (Imagine set notation, |A \intersect B| \divide |A \union B|)
def jaccard(a: set, b: set) -> float:
    if not a and not b:                          # Avoid division by 0
        return 1.0
    return len(a & b) / len(a | b)


# Parses a 'base' and 'other' file of edges (two different ACDC runs, hopefully of the same task under different metrics, thresholds or corruptions)
# Prints are self-explanatory given the above functions
def main():
    parser = argparse.ArgumentParser(description="Compare two ACDC edge.pkl outputs.")
    parser.add_argument("base", type=str, help="Baseline edges.pkl")
    parser.add_argument("other", type=str, help="Other edges.pkl (pass1/pass2)")
    parser.add_argument("--top", type=int, default=10, help="Show top-N new edges by score.")
    args = parser.parse_args()

    base_edges = load_edges(Path(args.base))
    other_edges = load_edges(Path(args.other))

    base_d = as_dict(base_edges)
    other_d = as_dict(other_edges)

    base_set = set(base_d.keys())
    other_set = set(other_d.keys())

    inter = base_set & other_set
    only_other = other_set - base_set
    only_base = base_set - other_set

    print("Base edges:", len(base_set))
    print("Other edges:", len(other_set))
    print("Overlap:", len(inter))
    print("Only in other:", len(only_other))
    print("Only in base:", len(only_base))
    print("Jaccard:", round(jaccard(base_set, other_set), 4))

    if only_other:
        ranked = sorted(only_other, key=lambda k: other_d[k], reverse=True)
        print(f"\nTop {min(args.top, len(ranked))} new edges in OTHER:")
        for k in ranked[: args.top]:
            
            print(f"  score={other_d[k]:.6f} edge={k}")

    if only_base:
        ranked = sorted(only_base, key=lambda k: base_d[k], reverse=True)
        print(f"\nTop {min(args.top, len(ranked))} edges missing from OTHER (present in BASE):")
        for k in ranked[: args.top]:
            print(f"  score={base_d[k]:.6f} edge={k}")


if __name__ == "__main__":
    main()
