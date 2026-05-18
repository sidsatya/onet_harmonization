"""Main entrypoint for running the full O*NET harmonization workflow.

This script orchestrates steps 1-4 and ensures output directories exist.
"""

import argparse

from pipeline_utils import ensure_dirs
from step1_prepare_task_statements import run_step1
from step2_cluster_task_statements import run_step2
from step3_harmonize_task_ratings import run_step3
from step4_build_final_outputs import run_step4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end O*NET harmonization pipeline.")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip step2 (embedding-based clustering).")
    parser.add_argument("--similarity-threshold", type=float, default=0.97)
    parser.add_argument("--k-neighbors", type=int, default=50)
    args = parser.parse_args()

    print("[pipeline] Starting O*NET harmonization pipeline")
    ensure_dirs()
    print("[pipeline] Step 1/4: Prepare task statements")
    run_step1()
    if not args.skip_clustering:
        print("[pipeline] Step 2/4: Cluster task statements")
        run_step2(similarity_threshold=args.similarity_threshold, k_neighbors=args.k_neighbors)
    else:
        print("[pipeline] Step 2/4: Skipped clustering")
    print("[pipeline] Step 3/4: Harmonize task ratings")
    run_step3()
    print("[pipeline] Step 4/4: Build final outputs")
    run_step4()
    print("[pipeline] Complete. Final files are in ./output")


if __name__ == "__main__":
    main()
