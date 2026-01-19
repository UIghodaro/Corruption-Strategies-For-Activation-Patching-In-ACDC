# inspect_pkl.py
import argparse
import pickle
from pathlib import Path
from typing import Any


# Summarise the contents of the pickle file
def summarise(obj: Any, max_items: int = 5, max_repr: int = 800) -> None:
    print("type:", type(obj))
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("dict keys (first 30):", keys[:30])
        # show a couple values briefly
        for k in keys[: min(3, len(keys))]:
            v = obj[k]
            print(f"  key={k!r}: type={type(v)}, repr={repr(v)[:200]}")
    elif isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        print("len:", len(seq))
        for i, item in enumerate(seq[:max_items]):
            print(f"[{i}]: type={type(item)}, repr={repr(item)[:200]}")
    else:
        print("repr:", repr(obj)[:max_repr])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect .pkl files (type/keys/preview).")
    
    # The pickle files that appear after runs of ACDC
    parser.add_argument("paths", nargs="*", default=["edges.pkl", "another_final_edges.pkl"], help='Inspect pickle files (can be a list of form: ["pickle1.pkl", "pickle2.pkl", ...])')
    
    # Top 'k' artefacts
    parser.add_argument("--max-items", type=int, default=5, help='Print the top X edges')
    
    args = parser.parse_args()

    for p_str in args.paths:
        p = Path(p_str)
        print("\n==", p, "==")
        if not p.exists():
            print("missing:", p)
            continue
        with p.open("rb") as f:
            obj = pickle.load(f)
        summarise(obj, max_items=args.max_items)


if __name__ == "__main__":
    main()
