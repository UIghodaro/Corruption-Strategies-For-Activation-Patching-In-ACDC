import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# This script is created to automate the entire 'running' part of the test pipeline, creating a summary of results

# Stream commands to the console to start a run, also save console outputs to a file 
def run(cmd, cwd=None, stdout_path=None):
    print("\n>>>", " ".join(cmd), "\n")
    
    if stdout_path is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

# Auto git commits 
def safe_git_info(repo_root: Path):
    def _try(args):
        try:
            out = subprocess.check_output(args, cwd=repo_root, text=True).strip()
            return out
        except Exception:
            return None

    return {
        "commit": _try(["git", "rev-parse", "HEAD"]),
        "status": _try(["git", "status", "--porcelain"]),
        "branch": _try(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }

def python_env_info():
    info = {
        "python_exe": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    # torch/cuda info if available
    try:
        import torch
        info.update({
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
        })
    except Exception as e:
        info["torch_import_error"] = repr(e)
    return info

def load_edges_count(run_dir: Path):
    """Count edges in the final circuit (another_final_edges.pkl). (Doesn't fail the run if missing.)"""
    import pickle
    p = run_dir / "another_final_edges.pkl"
    if not p.exists():
        return None
    try:
        obj = pickle.load(open(p, "rb"))
        # another_final_edges.pkl is a list of (edge_key, score) tuples for the completed circuit
        return len(obj)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="docstring")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--corrupted-batch-size", type=int, default=4)
    ap.add_argument("--seeds", default="0,1", help="comma-separated seeds, e.g. 0,1")
    ap.add_argument("--metric", default=None, help="")
    ap.add_argument("--extra", default="", help="extra flags to be passed through to acdc.main")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_root = repo_root / "runs"
    runs_root.mkdir(exist_ok=True)


    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Docstring corruption variants (dataset_version keys from docstring_induction_prompt_generator).
    # Comment out individual entries to run a subset.
    DOCSTRING_VERSIONS = [
        "random_random",               # default; both def and doc args fully randomised
        "random_doc",                  # doc arg names replaced; def intact
        "random_def",                  # def arg names replaced; doc intact
        "random_answer",               # only the target answer arg in def replaced
        "random_def_doc",              # random_def + random_doc combined
        "random_answer_doc",           # random_answer + random_doc combined
        "vary_length_doc_desc",        # arg names preserved; description words redistributed
        "vary_length_doc_desc_random_doc",  # description redistributed + random_doc
    ]

    # If a condition doesn't work, comment it out and rerun.
    conditions = (
        [("zero_ablation", ["--zero-ablation"])]
        + [(v, ["--dataset-version", v]) for v in DOCSTRING_VERSIONS]
    )

    summary_rows = []

    # Concurrently run corruption strategies under different seeds
    for seed in seeds:
        for (cond_name, cond_flags) in conditions:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{ts}_{args.task}_{cond_name}_t{args.threshold}_s{seed}_{args.device}"
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Write metadata
            config = {
                "task": args.task,
                "device": args.device,
                "threshold": args.threshold,
                "corrupted_batch_size": args.corrupted_batch_size,
                "seed": seed,
                "metric": args.metric,
                "cond_name": cond_name,
                "cond_flags": cond_flags,
                "extra": args.extra,
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (run_dir / "env.json").write_text(json.dumps(python_env_info(), indent=2), encoding="utf-8")
            (run_dir / "git.json").write_text(json.dumps(safe_git_info(repo_root), indent=2), encoding="utf-8")

            # Build ACDC.main command for console
            cmd = [
                sys.executable, "-m", "acdc.main",
                "--task", args.task,
                "--threshold", str(args.threshold),
                "--device", args.device,
                "--corrupted-batch-size", str(args.corrupted_batch_size),
                "--seed", str(seed),
            ]

            if args.metric:
                cmd += ["--metric", args.metric]

            cmd += cond_flags

            # Pass-through extra flags (string)
            if args.extra.strip():
                cmd += args.extra.strip().split()

            # Run and log
            stdout_path = run_dir / "stdout.txt"
            try:
                run(cmd, cwd=repo_root, stdout_path=stdout_path)
            except Exception as e:
                # If a row fails, record it
                (run_dir / "error.txt").write_text(repr(e), encoding="utf-8")

            # Copy expected artefacts into run_dir if they were created elsewhere
            # If ACDC already writes into a run folder, skip this section later.
            for fname in ["edges.pkl", "another_final_edges.pkl"]:
                src = repo_root / fname
                if src.exists():
                    dst = run_dir / fname
                    try:
                        dst.write_bytes(src.read_bytes())
                    except Exception:
                        pass

            edge_count = load_edges_count(run_dir)

            summary_rows.append({
                "run_dir": str(run_dir),
                "task": args.task,
                "condition": cond_name,
                "seed": seed,
                "threshold": args.threshold,
                "device": args.device,
                "corrupted_batch_size": args.corrupted_batch_size,
                "edges_count": edge_count,
                "status": "ok" if not (run_dir / "error.txt").exists() else "error",
            })

    # Write suite summary
    summary_path = runs_root / f"SUITE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.task}.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("\nSuite summary:", summary_path)

if __name__ == "__main__":
    main()
