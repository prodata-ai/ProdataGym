"""
Episode logger — records every training step to build the Pro verifier dataset.

The logged data is the raw material for:
  1. Detecting new reward-hacking patterns
  2. Training the Pro verifier's gaming classifier
  3. Building the gaming pattern database

Usage:
    from prodata.core.episode_logger import EpisodeLogger
    logger = EpisodeLogger()   # uses PRODATA_LOG_DIR env var or ./logs/

    # In your training loop:
    logger.log_step(
        task_id=info["task_id"],
        action_code=code,
        sim_outputs=info["dimension_scores"],
        reward=reward,
        passed=info["success"],
        customer_id=os.getenv("PRODATA_CUSTOMER_ID"),
    )

Set PRODATA_LOG_DIR to a shared path (S3 mount, NFS, etc.) to aggregate
logs from multiple training runs into a single dataset.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeStep:
    """One agent step — the atomic unit of the training dataset."""
    step_id:       str              # unique ID for deduplication
    timestamp:     float            # unix timestamp
    customer_id:   str | None       # set via PRODATA_CUSTOMER_ID env var
    domain:        str              # "cad", "solar", "rf"
    task_id:       str
    episode_step:  int              # step number within episode (0-indexed)
    action_code:   str              # the CadQuery code the agent generated
    sim_outputs:   dict[str, Any]   # all simulator outputs (SF, cost, bbox, etc.)
    dimension_scores: dict[str, float]  # {"structural": 0.8, "cost": 0.6, ...}
    reward:        float
    passed:        bool
    # Optional metadata for detecting gaming patterns
    safety_factor:    float | None = None
    cost_usd:         float | None = None
    bounding_box_mm:  list[float] | None = None
    model_name:       str | None = None   # which LLM generated the code
    # Gaming flags — set by Pro verifier when available
    gaming_detected:  bool = False
    gaming_pattern:   str | None = None   # e.g. "thin_tall_box", "hollow_shell"


class EpisodeLogger:
    """
    Writes episode steps to newline-delimited JSON (JSONL) files.

    File layout:
        {log_dir}/
            {domain}/
                {date}/
                    {customer_id}_{uuid}.jsonl   # one file per training run

    Each line in the JSONL file is one EpisodeStep, serialized to JSON.
    Files are flushed after every write (safe for crash recovery).

    To aggregate for Pro verifier training:
        cat logs/cad/2026-*/**.jsonl | sort -k2 > full_dataset.jsonl
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        domain: str = "cad",
        customer_id: str | None = None,
        model_name: str | None = None,
        enabled: bool = True,
    ):
        """
        Args:
            log_dir:     Where to write logs. Defaults to $PRODATA_LOG_DIR or ./logs/
            domain:      Which gym domain ("cad", "solar", "rf")
            customer_id: Tag logs by customer. Defaults to $PRODATA_CUSTOMER_ID.
            model_name:  Optional — which LLM is generating actions.
            enabled:     Set False to disable logging without changing calling code.
        """
        self.enabled = enabled
        if not enabled:
            return

        self.domain = domain
        self.customer_id = customer_id or os.getenv("PRODATA_CUSTOMER_ID")
        self.model_name = model_name

        base = log_dir or os.getenv("PRODATA_LOG_DIR") or Path("logs")
        date_str = time.strftime("%Y-%m-%d")
        run_id = uuid.uuid4().hex[:8]
        cid = self.customer_id or "anon"

        log_path = Path(base) / domain / date_str
        log_path.mkdir(parents=True, exist_ok=True)

        filename = f"{cid}_{run_id}.jsonl"
        self._file = open(log_path / filename, "a", buffering=1)  # line-buffered
        self._path = log_path / filename

    def log_step(
        self,
        task_id: str,
        action_code: str,
        sim_outputs: dict[str, Any],
        dimension_scores: dict[str, float],
        reward: float,
        passed: bool,
        episode_step: int = 0,
        gaming_detected: bool = False,
        gaming_pattern: str | None = None,
    ) -> None:
        """
        Log one agent step. Call this after every env.step() in your training loop.

        Args:
            task_id:          From info["task_id"]
            action_code:      The Python string the agent submitted
            sim_outputs:      Raw simulator outputs (from info or sim_result.outputs)
            dimension_scores: From info["dimension_scores"]
            reward:           The float reward returned by env.step()
            passed:           From info["success"]
            episode_step:     From obs["step"] or info["step"]
            gaming_detected:  From info["gaming_detected"] (Pro verifier only)
            gaming_pattern:   Specific pattern name if gaming was detected
        """
        if not self.enabled:
            return

        step = EpisodeStep(
            step_id=uuid.uuid4().hex,
            timestamp=time.time(),
            customer_id=self.customer_id,
            domain=self.domain,
            task_id=task_id,
            episode_step=episode_step,
            action_code=action_code,
            sim_outputs=sim_outputs,
            dimension_scores=dimension_scores,
            reward=reward,
            passed=passed,
            safety_factor=sim_outputs.get("safety_factor"),
            cost_usd=sim_outputs.get("total_cost_usd"),
            bounding_box_mm=sim_outputs.get("bounding_box_mm"),
            model_name=self.model_name,
            gaming_detected=gaming_detected,
            gaming_pattern=gaming_pattern,
        )
        self._file.write(json.dumps(asdict(step)) + "\n")

    def close(self) -> None:
        """Call when training run ends. Safe to call multiple times."""
        if self.enabled and hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def log_path(self) -> Path | None:
        """Path to the current JSONL file, for reference."""
        return getattr(self, "_path", None)

    # ------------------------------------------------------------------
    # Offline utilities — call these outside of training loops
    # ------------------------------------------------------------------

    @staticmethod
    def iter_steps(log_dir: str | Path, domain: str = "cad"):
        """
        Iterate over all logged steps across all dates and customers.

        Yields EpisodeStep-like dicts (from JSON). Use for building datasets.

        Example:
            for step in EpisodeLogger.iter_steps("logs/"):
                if step["passed"] and step["safety_factor"] > 20:
                    print("Suspicious:", step["task_id"], step["safety_factor"])
        """
        base = Path(log_dir) / domain
        for jsonl_file in sorted(base.glob("**/*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    @staticmethod
    def build_gaming_dataset(
        log_dir: str | Path,
        domain: str = "cad",
        sf_threshold: float = 20.0,
        cost_score_threshold: float = 0.2,
    ) -> list[dict]:
        """
        Extract likely-gamed episodes for Pro verifier training.

        A step is flagged as gaming if:
          - It passed (reward >= 0.7)
          - Safety factor is suspiciously high (> sf_threshold)
          - Cost score is low (model inflated height, wasted material)

        Returns a list of dicts with fields: task_id, action_code, sf, cost_score.
        These are labeled "gaming" examples for the Pro verifier classifier.
        """
        gaming = []
        for step in EpisodeLogger.iter_steps(log_dir, domain):
            sf = step.get("safety_factor") or 0
            cost_score = step.get("dimension_scores", {}).get("cost", 1.0)
            if step["passed"] and sf > sf_threshold and cost_score < cost_score_threshold:
                gaming.append({
                    "task_id":     step["task_id"],
                    "action_code": step["action_code"],
                    "safety_factor": sf,
                    "cost_score":  cost_score,
                    "reward":      step["reward"],
                    "label":       "gaming",
                })
        return gaming

    @staticmethod
    def summary(log_dir: str | Path, domain: str = "cad") -> dict:
        """
        Print a quick summary of logged data.

        Returns:
            {
                "total_steps": int,
                "unique_customers": int,
                "pass_rate": float,
                "mean_reward": float,
                "suspected_gaming_rate": float,
            }
        """
        total = 0
        passed = 0
        rewards = []
        customers = set()
        gaming_count = 0

        for step in EpisodeLogger.iter_steps(log_dir, domain):
            total += 1
            if step["passed"]:
                passed += 1
            rewards.append(step["reward"])
            if step.get("customer_id"):
                customers.add(step["customer_id"])
            sf = step.get("safety_factor") or 0
            cost_score = step.get("dimension_scores", {}).get("cost", 1.0)
            if step["passed"] and sf > 20 and cost_score < 0.2:
                gaming_count += 1

        return {
            "total_steps":           total,
            "unique_customers":      len(customers),
            "pass_rate":             round(passed / total, 3) if total else 0,
            "mean_reward":           round(sum(rewards) / len(rewards), 3) if rewards else 0,
            "suspected_gaming_rate": round(gaming_count / total, 3) if total else 0,
        }
