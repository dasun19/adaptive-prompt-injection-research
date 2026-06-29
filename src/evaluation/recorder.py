"""
recorder.py
───────────
Records every experiment run to JSONL.
One line per run — works for baseline, attack, and defense runs.

Data stored per run:
  - Run metadata     (id, timestamp, architecture, attack type, defense)
  - Agent traces     (hop, role, input, output, is_compromised, time)
  - Metrics          (per_hop_compromised, propagation_depth, ASR, latency)

File locations:
    experiments/results/devops_pipeline/sequential_runs.jsonl
    experiments/results/devops_pipeline/hierarchical_runs.jsonl

Load in pandas:
  import pandas as pd
    df = pd.read_json("experiments/results/devops_pipeline/sequential_runs.jsonl", lines=True)
"""

import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Recorder no longer performs compromise detection; verdicts are supplied by runner


class RunRecorder:
    """
    Records one experiment run at a time.

    Typical usage:
        recorder = RunRecorder()

        run = recorder.new_run(architecture="sequential", ...)
        # ... run the workflow ...
        for i, task_output in enumerate(result.tasks_output):
            recorder.record_agent(run, hop=i+1, ...)
        run = recorder.finish(run)
        recorder.print_summary(run)
    """

    def __init__(self, results_dir: str = "experiments/results/devops_pipeline"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Start a run ───────────────────────────────────────────────────────

    def new_run(
        self,
        architecture:     str,
        task_scenario:    str,
        threat:           str,
        attack_type:      str            = "none",
        injection_point:  Optional[str]  = None,
        injection_text:   Optional[str]  = None,
        injection_goal:   Optional[str]  = None,
        defense:          str            = "none",
        defense_strategy: str            = "none",
    ) -> Dict:
        """
        Initialise a new run record.

        Args:
            architecture     : "sequential" | "hierarchical"
            task_scenario    : e.g. "cyber_threat_intel"
            threat           : the threat string passed to agents
            attack_type      : "none" | "direct_override" | "indirect_injection"
                               | "prompt_infection" | "typographic"
            injection_point  : which hop the injection enters  e.g. "hop_1"
            injection_text   : the full injection string
            injection_goal   : e.g. "exfiltrate_data"
            defense          : "none" | "lakera" | "aws_bedrock" | "azure"
                               | "deberta" | "prompt_guard" | "promptarmor"
            defense_strategy : "none" | "input_only" | "hop_by_hop"
        """
        return {
            # ── identifiers ─────────────────────────────────────────────────
            "run_id":           str(uuid.uuid4()),
            "timestamp":        datetime.now().isoformat(),

            # ── experiment config ────────────────────────────────────────────
            "architecture":     architecture,
            "task_scenario":    task_scenario,
            "threat":           threat,
            "llm":              "llama-3.1-70b-versatile",
            "framework":        "langgraph",

            # ── attack config ────────────────────────────────────────────────
            "attack_type":      attack_type,
            "is_baseline":      attack_type == "none",
            "injection_point":  injection_point,
            "injection_text":   injection_text,
            "injection_goal":   injection_goal,

            # ── defense config ───────────────────────────────────────────────
            "defense":          defense,
            "defense_strategy": defense_strategy,

            # ── internal timer (removed before saving) ───────────────────────
            "_start_time":      time.time(),

            # ── per-agent data ───────────────────────────────────────────────
            "agent_traces":     [],
        }

    # ── 2. Record each agent's output ────────────────────────────────────────

    def record_agent(
        self,
        run:              Dict,
        hop:              int,
        agent_role:       str,
        task_description: str,
        output:           str,
        execution_time:   Optional[float] = None,
        input_prompt:     Optional[str] = None,
        is_compromised:   Optional[bool] = None,
        keyword_verdict:  Optional[bool] = None,
        llm_verdict:      Optional[bool] = None,
    ) -> None:
        """Append an agent hop record to `run["agent_traces"]`.

        This method no longer runs compromise detection; it persists
        the fields provided by the caller (runner post-processing).
        """

        run["agent_traces"].append({
            "hop":                  hop,
            "agent_role":           agent_role,
            "task_description":     task_description[:300],
            "input_prompt":         input_prompt,
            "output":               output,
            "output_length_chars":  len(output),
            "is_compromised":       bool(is_compromised) if is_compromised is not None else False,
            "keyword_verdict":      bool(keyword_verdict) if keyword_verdict is not None else False,
            "llm_verdict":          bool(llm_verdict) if llm_verdict is not None else False,
            "execution_time_sec":   round(execution_time, 3) if execution_time else None,
        })

    # ── 3. Finish and save ───────────────────────────────────────────────────

    def finish(self, run: Dict) -> Dict:
        """
        Compute final metrics and append the run to the JSONL file.
        """
        # If the runner already populated metrics (preferred), respect them.
        traces = run.get("agent_traces", [])
        metrics = run.get("metrics", None)

        if not metrics or "total_execution_time_sec" not in metrics:
            total_time = round(time.time() - run.pop("_start_time"), 2)
            per_hop = [t.get("is_compromised", False) for t in traces]
            metrics = {
                "per_hop_compromised": per_hop,
                "propagation_depth": sum(per_hop),
                "attack_success": per_hop[-1] if per_hop else False,
                "total_execution_time_sec": total_time,
                "analysis_overhead_sec": 0.0,
                "per_hop_execution_time_sec": [t.get("execution_time_sec") for t in traces],
            }

        # Ensure derived fields exist
        metrics.setdefault("analysis_overhead_sec", 0.0)
        metrics.setdefault("per_hop_execution_time_sec", [t.get("execution_time_sec") for t in traces])
        metrics.setdefault("per_hop_compromised", [t.get("is_compromised", False) for t in traces])
        metrics.setdefault("propagation_depth", sum(metrics["per_hop_compromised"]))
        metrics.setdefault("attack_success", metrics["per_hop_compromised"][-1] if metrics["per_hop_compromised"] else False)

        run["metrics"] = metrics

        # Save to JSONL
        out = self.results_dir / f"{run['architecture']}_runs.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")

        return run

    # ── 4. Print summary ─────────────────────────────────────────────────────

    def print_summary(self, run: Dict) -> None:
        m    = run["metrics"]
        hops = " → ".join(
            f"hop{i+1}:{'✓' if c else '✗'}"
            for i, c in enumerate(m.get("per_hop_compromised", []))
        )
        print(f"\n{'─'*55}")
        print(f"  run_id     : {run['run_id'][:12]}...")
        print(f"  arch       : {run['architecture']}")
        print(f"  baseline   : {run['is_baseline']}")
        print(f"  attack     : {run['attack_type']}")
        print(f"  defense    : {run['defense']} / {run['defense_strategy']}")
        print(f"  hops       : {hops}")
        print(f"  depth      : {m.get('propagation_depth')} / {len(m.get('per_hop_compromised', []))}")
        print(f"  success    : {m.get('attack_success')}")
        print(f"  time       : {m.get('total_execution_time_sec')}s")
        print(f"{'─'*55}\n")
