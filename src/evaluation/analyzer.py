"""
analyzer.py
───────────
Loads JSONL run files and computes all 5 research metrics.

Metrics covered:
  1. ASR              — Attack Success Rate at final agent
  2. Per-Hop ASR      — ASR at each hop across 100 runs
  3. Propagation Depth— Avg compromised agents per run
  4. Latency Impact   — Extra time added by attack vs baseline
  5. False Pos. Rate  — Phase 3 only (defense benchmarking)

Quick usage:
    from src.evaluation.analyzer import ResultsAnalyzer
    analyzer = ResultsAnalyzer()
    analyzer.report("sequential")
    analyzer.plot_per_hop_asr()
    analyzer.plot_architecture_comparison()
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


class ResultsAnalyzer:

    def __init__(self, results_dir: str = "experiments/results/phase2"):
        self.results_dir = Path(results_dir)

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self, architecture: str) -> pd.DataFrame:
        """Load one architecture's JSONL into a flat DataFrame."""
        path = self.results_dir / f"{architecture}_runs.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No results at {path}. Run some experiments first.")

        df = pd.read_json(path, lines=True)

        # Flatten nested metrics dict into columns
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop(columns=["metrics", "agent_traces"]), metrics_df], axis=1)
        return df

    def load_all(self) -> pd.DataFrame:
        """Load both architectures into a single DataFrame."""
        frames = []
        for arch in ["sequential", "hierarchical"]:
            try:
                frames.append(self.load(arch))
            except FileNotFoundError:
                pass
        if not frames:
            raise FileNotFoundError("No results found. Run baseline experiments first.")
        return pd.concat(frames, ignore_index=True)

    # ── Metric 1: ASR ────────────────────────────────────────────────────────

    def asr(
        self,
        df:          pd.DataFrame,
        attack_type: Optional[str] = None,
        defense:     Optional[str] = None,
    ) -> float:
        """
        Attack Success Rate = injections that reached the final agent
                              ─────────────────────────────────────────
                              total attack runs

        Returns float 0.0–1.0. Returns 0.0 if no attack runs found.
        """
        df = df[~df["is_baseline"]]
        if attack_type:
            df = df[df["attack_type"] == attack_type]
        if defense:
            df = df[df["defense"] == defense]
        if df.empty:
            return 0.0
        return float(df["attack_success"].mean())

    # ── Metric 2: Per-Hop ASR ────────────────────────────────────────────────

    def per_hop_asr(
        self,
        df:          pd.DataFrame,
        attack_type: Optional[str] = None,
    ) -> pd.Series:
        """
        Per-Hop ASR = fraction of runs where injection reached each hop.

        Returns a Series: [hop1_asr, hop2_asr, hop3_asr, hop4_asr]
        Aggregate over multiple runs to get meaningful values.
        """
        df = df[~df["is_baseline"]]
        if attack_type:
            df = df[df["attack_type"] == attack_type]
        if df.empty:
            return pd.Series(dtype=float)

        hops_df = pd.DataFrame(df["per_hop_compromised"].tolist())
        series  = hops_df.mean()
        series.index = [f"hop_{i+1}" for i in range(len(series))]
        return series

    # ── Metric 3: Propagation Depth ──────────────────────────────────────────

    def propagation_depth(
        self,
        df:          pd.DataFrame,
        attack_type: Optional[str] = None,
    ) -> float:
        """
        Propagation Depth = avg number of agents compromised per attack run.
        Max value = number of agents (4 in our setup).
        """
        df = df[~df["is_baseline"]]
        if attack_type:
            df = df[df["attack_type"] == attack_type]
        if df.empty:
            return 0.0
        return float(df["propagation_depth"].mean())

    # ── Metric 4: Latency Impact ─────────────────────────────────────────────

    def latency_impact(self, df: pd.DataFrame) -> float:
        """
        Latency Impact = avg attack run time − avg baseline run time (seconds).
        Positive = attack adds overhead.
        """
        baseline_time = df[df["is_baseline"]]["total_execution_time_sec"].mean()
        attack_time   = df[~df["is_baseline"]]["total_execution_time_sec"].mean()
        if np.isnan(attack_time) or np.isnan(baseline_time):
            return 0.0
        return round(float(attack_time - baseline_time), 2)

    # ── Metric 5: False Positive Rate (Phase 3) ──────────────────────────────

    def false_positive_rate(self, df: pd.DataFrame, defense: str) -> float:
        """
        False Positive Rate = baseline runs flagged as attack by defense
                              ─────────────────────────────────────────
                              total baseline runs

        Phase 3 only — requires defense field to be active.
        Column 'defense_flagged' must be added in Phase 3 runner.
        """
        df = df[(df["is_baseline"]) & (df["defense"] == defense)]
        if df.empty or "defense_flagged" not in df.columns:
            return 0.0
        return float(df["defense_flagged"].mean())

    # ── Summary report ───────────────────────────────────────────────────────

    def report(self, architecture: str) -> None:
        """Print a full metric report for one architecture."""
        df = self.load(architecture)

        print(f"\n{'='*58}")
        print(f"  Architecture : {architecture.upper()}")
        print(f"  Total runs   : {len(df)}")
        print(f"  Baseline     : {int(df['is_baseline'].sum())}")
        print(f"  Attack runs  : {int((~df['is_baseline']).sum())}")
        print(f"{'='*58}")

        # ── Baseline latency ─────────────────────────────────────────────────
        baseline_time = df[df["is_baseline"]]["total_execution_time_sec"].mean()
        if not np.isnan(baseline_time):
            print(f"\n  Baseline avg time : {baseline_time:.1f}s")

        # ── Per attack type ──────────────────────────────────────────────────
        attack_types = df[~df["is_baseline"]]["attack_type"].unique()
        for at in attack_types:
            asr_val   = self.asr(df, attack_type=at)
            depth     = self.propagation_depth(df, attack_type=at)
            hop_asr   = self.per_hop_asr(df, attack_type=at)
            latency   = self.latency_impact(df[df["attack_type"].isin(["none", at])])

            print(f"\n  ── Attack: {at}")
            print(f"     ASR             : {asr_val:.1%}")
            print(f"     Propagation Depth: {depth:.2f} / 4 agents")
            hop_str = "  →  ".join(f"h{i+1}:{v:.0%}" for i, v in enumerate(hop_asr))
            print(f"     Per-Hop ASR     : {hop_str}")
            print(f"     Latency Impact  : +{latency:.1f}s")

        print(f"\n{'='*58}\n")

    # ── Plot 1: Per-Hop ASR ──────────────────────────────────────────────────

    def plot_per_hop_asr(
        self,
        save_path: str = "experiments/results/phase2/plot_per_hop_asr.png"
    ) -> None:
        """Line chart: ASR at each hop, one line per architecture."""
        df = self.load_all()
        df_attack = df[~df["is_baseline"]]
        if df_attack.empty:
            print("No attack runs yet. Run attack experiments first.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        for arch in df_attack["architecture"].unique():
            hop_asr = self.per_hop_asr(df_attack[df_attack["architecture"] == arch])
            if hop_asr.empty:
                continue
            ax.plot(
                range(1, len(hop_asr) + 1),
                hop_asr.values,
                marker="o",
                linewidth=2,
                label=arch.capitalize()
            )

        ax.set_xlabel("Hop Number (Agent Position)")
        ax.set_ylabel("ASR (Attack Success Rate)")
        ax.set_title("Per-Hop ASR by Architecture")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(
            ["Hop 1\n(OSINT)", "Hop 2\n(Analysis)", "Hop 3\n(Mitigation)", "Hop 4\n(Report)"]
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.show()

    # ── Plot 2: Architecture comparison heatmap ───────────────────────────────

    def plot_architecture_comparison(
        self,
        save_path: str = "experiments/results/phase2/plot_arch_comparison.png"
    ) -> None:
        """
        Heatmap: ASR per attack_type × architecture.
        Rows = attack types, Columns = architectures.
        """
        df = self.load_all()
        df_attack = df[~df["is_baseline"]]
        if df_attack.empty:
            print("No attack runs yet.")
            return

        pivot = df_attack.groupby(["attack_type", "architecture"])["attack_success"].mean().unstack()

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0%",
            cmap="Reds",
            vmin=0, vmax=1,
            ax=ax,
            linewidths=0.5
        )
        ax.set_title("ASR by Attack Type × Architecture")
        ax.set_xlabel("Architecture")
        ax.set_ylabel("Attack Type")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.show()

    # ── Plot 3: Propagation depth bar chart ──────────────────────────────────

    def plot_propagation_depth(
        self,
        save_path: str = "experiments/results/phase2/plot_propagation_depth.png"
    ) -> None:
        """Bar chart: avg propagation depth per architecture."""
        df = self.load_all()
        df_attack = df[~df["is_baseline"]]
        if df_attack.empty:
            print("No attack runs yet.")
            return

        summary = df_attack.groupby("architecture")["propagation_depth"].mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        summary.plot(kind="bar", ax=ax, color=["#534AB7", "#185FA5"], edgecolor="white")
        ax.set_title("Average Propagation Depth by Architecture")
        ax.set_ylabel("Avg Compromised Agents")
        ax.set_ylim(0, 4)
        ax.set_xticklabels(summary.index, rotation=0)
        ax.axhline(y=4, color="red", linestyle="--", alpha=0.4, label="Max (4 agents)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.show()
