"""
main.py
───────
Run DevOps-pipeline experiments from the command line.

Baseline examples:
    python main.py --arch sequential   --project "payments-service PR #482"
    python main.py --arch hierarchical --project "payments-service PR #482"

Results saved to:
    experiments/results/devops_pipeline/sequential_runs.jsonl
    experiments/results/devops_pipeline/hierarchical_runs.jsonl
"""

import argparse
from src.evaluation.recorder            import RunRecorder
from src.agents.devops_pipeline.runner   import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="DevOps Pipeline — Multi-Agent Experiment Runner"
    )
    parser.add_argument(
        "--arch",
        choices=["sequential", "hierarchical"],
        required=True,
        help="Agent architecture to use"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="payments-service PR #482",
        help="Project or pull request label to investigate"
    )
    parser.add_argument(
        "--pr-content",
        type=str,
        default="Adds retry logic to the payment webhook handler.",
        help="Pull request description and diff text"
    )
    parser.add_argument(
        "--test-logs",
        type=str,
        default="42 passed, 0 failed. Coverage: 87%.",
        help="Raw CI/test log text"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat this run (default: 1)"
    )
    args = parser.parse_args()

    recorder = RunRecorder()

    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n{'='*55}")
            print(f"  Run {i+1} of {args.runs}")
            print(f"{'='*55}")

        run_experiment(
            architecture  = args.arch,
            project       = args.project,
            pr_content    = args.pr_content,
            test_logs     = args.test_logs,
            task_scenario = "devops_release_pipeline",
            attack_type   = "none",       # baseline — no injection
            recorder      = recorder,
        )

    print(f"\n✓ Results saved to:")
    print(f"  experiments/results/devops_pipeline/{args.arch}_runs.jsonl\n")


if __name__ == "__main__":
    main()
