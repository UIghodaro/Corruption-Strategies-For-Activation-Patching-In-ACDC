# save_run_artifacts.py
import argparse
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_ARTIFACTS = [
    "edges.pkl",
    "another_final_edges.pkl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Move run artifacts into a timestamped folder.")
    parser.add_argument("--outdir", default="runs", help="Base directory for saved runs.")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag to append to the timestamp folder name (e.g. tracr-proportion_base).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_ARTIFACTS,
        help="Artifact filenames to move (relative to current working dir).",
    )
    parser.add_argument("--copy", action="store_true", help="Copy instead of move.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = stamp if not args.tag else f"{stamp}_{args.tag}"
    out_base = Path(args.outdir)
    out_dir = out_base / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for fname in args.files:
        src = Path(fname)
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        dst = out_dir / src.name
        if args.copy:
            shutil.copy2(src, dst)
            print(f"copied: {src} -> {dst}")
        else:
            shutil.move(str(src), str(dst))
            print(f"moved: {src} -> {dst}")
        moved_any = True

    if not moved_any:
        print("No artifacts found to save.")
    else:
        print(f"Saved artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
