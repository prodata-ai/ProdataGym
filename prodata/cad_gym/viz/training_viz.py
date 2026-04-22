"""
TrainingVisualizer — live gamified dashboard for Colab RL training.

Works entirely off the standard env.step() return values:
    obs, reward, terminated, truncated, info = env.step(code)

Keys used:
    info["dimension_scores"]  → {"structural": float, "cost": float, "geometry": float}
    info["success"]           → bool
    info["warnings"]          → list[str]
    info["mesh_file"]         → str | None  (path to generated STL)
    info["task_id"]           → str
    obs["safety_factor"]      → float32[1]
    obs["cost_usd"]           → float32[1]

Dependencies (all available on Colab):
    plotly, trimesh, matplotlib, numpy, IPython
"""

from __future__ import annotations

import numpy as np
from collections import deque
from IPython.display import display, clear_output, HTML


# ── ACHIEVEMENT DEFINITIONS ─────────────────────────────────────────────────

ACHIEVEMENTS = {
    "first_pass":       ("🥇", "First Pass!",           "First design to pass verification"),
    "three_streak":     ("🔥", "On Fire!",               "3 passing designs in a row"),
    "five_streak":      ("⚡", "Hot Streak!",            "5 passing designs in a row"),
    "under_budget":     ("💰", "Penny Pincher",          "Cost score ≥ 0.95"),
    "structurally_s":   ("🏗️",  "Structurally Sound",    "Structural score ≥ 0.90"),
    "perfect_geometry": ("📐", "Perfect Fit",            "Geometry score = 1.0 three times"),
    "comeback":         ("🔄", "Comeback Kid",           "Pass after 10+ consecutive failures"),
    "half_century":     ("5️⃣0️⃣", "Half Century",         "50 episodes completed"),
    "century":          ("💯", "Century",                "100 episodes completed"),
    "beat_baseline":    ("📈", "Beat the Baseline",      "Rolling pass rate > 14% (zero-shot baseline)"),
    "research_grade":   ("🎓", "Research Grade",         "Rolling pass rate ≥ 50%"),
}


class TrainingVisualizer:
    """
    Drop-in dashboard for Colab RL training loops.

    Usage:
        viz = TrainingVisualizer(target_pass_rate=0.67)

        for episode in range(N):
            obs, info = env.reset()
            code = model.generate(...)
            obs, reward, terminated, truncated, info = env.step(code)

            viz.update(episode, obs, reward, info, code)
            if episode % 10 == 0:
                viz.render()

        viz.final_summary()
    """

    def __init__(
        self,
        target_pass_rate: float = 0.67,
        rolling_window: int = 50,
        viz_every: int = 10,
    ):
        self.target_pass_rate = target_pass_rate
        self.rolling_window   = rolling_window
        self.viz_every        = viz_every

        # Episode history
        self.episodes:          list[int]   = []
        self.rewards:           list[float] = []
        self.passed:            list[bool]  = []
        self.structural_scores: list[float] = []
        self.cost_scores:       list[float] = []
        self.geometry_scores:   list[float] = []
        self.safety_factors:    list[float] = []
        self.costs_usd:         list[float] = []

        # Best design tracking
        self.best: dict = {"reward": -999, "episode": None, "mesh_file": None,
                           "dim_scores": {}, "task_id": None}

        # Gamification state
        self.unlocked_achievements: set[str] = set()
        self.streak: int = 0
        self.fail_streak: int = 0
        self._perfect_geom_count: int = 0

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def update(
        self,
        episode: int,
        obs: dict,
        reward: float,
        info: dict,
        code: str | None = None,
    ) -> list[str]:
        """
        Record one episode step. Returns list of newly unlocked achievement keys.
        Call this every episode. Call render() as often as you want.
        """
        dim = info.get("dimension_scores", {})
        sf  = float(obs.get("safety_factor", [-1.0])[0])
        c   = float(obs.get("cost_usd",      [-1.0])[0])

        self.episodes.append(episode)
        self.rewards.append(reward)
        self.passed.append(info.get("success", False))
        self.structural_scores.append(dim.get("structural", 0.0))
        self.cost_scores.append(dim.get("cost", 0.0))
        self.geometry_scores.append(dim.get("geometry", 0.0))
        self.safety_factors.append(sf if sf > 0 else 0.0)
        self.costs_usd.append(c if c > 0 else 0.0)

        if info.get("success") and reward > self.best["reward"]:
            self.best = {
                "reward":     reward,
                "episode":    episode,
                "mesh_file":  info.get("mesh_file"),
                "dim_scores": dim,
                "task_id":    info.get("task_id"),
            }

        # Streak tracking
        if info.get("success"):
            self.streak += 1
            self.fail_streak = 0
        else:
            if self.fail_streak >= 10 and info.get("success"):
                self._unlock("comeback")
            self.fail_streak += 1
            self.streak = 0

        return self._check_achievements(episode, dim)

    def render(self, episode: int | None = None) -> None:
        """
        Render the full dashboard. Safe to call every episode — uses clear_output.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly not installed — run: pip install plotly")
            return

        clear_output(wait=True)
        ep = episode or (self.episodes[-1] if self.episodes else 0)
        n  = len(self.episodes)

        # ── Header ──────────────────────────────────────────────────────────
        rolling_pass = self._rolling_pass_rate()
        bar_filled   = int(rolling_pass * 20)
        bar          = "█" * bar_filled + "░" * (20 - bar_filled)
        target_pct   = int(self.target_pass_rate * 100)
        streak_str   = f"🔥 {self.streak}" if self.streak >= 3 else str(self.streak)

        print(f"\n{'━'*68}")
        print(f"  🎮  CAD-GYM DASHBOARD   Episode {ep}   Target: {target_pct}% pass rate")
        print(f"{'━'*68}")
        print(f"  Pass rate (last {self.rolling_window}):  [{bar}] {rolling_pass:.0%}  "
              f"{'✅' if rolling_pass >= self.target_pass_rate else '⏳'}")
        print(f"  Avg reward (last {self.rolling_window}): {self._rolling_mean(self.rewards):.3f}   "
              f"Success streak: {streak_str}")
        print(f"{'━'*68}\n")

        # ── Achievements ────────────────────────────────────────────────────
        if self.unlocked_achievements:
            icons = "  ".join(
                ACHIEVEMENTS[k][0] + " " + ACHIEVEMENTS[k][1]
                for k in self.unlocked_achievements
            )
            print(f"  Achievements: {icons}\n")

        # ── Last design scores ───────────────────────────────────────────────
        if self.structural_scores:
            s  = self.structural_scores[-1]
            c  = self.cost_scores[-1]
            g  = self.geometry_scores[-1]
            r  = self.rewards[-1]
            ok = self.passed[-1]

            print(f"  {'✅ PASS' if ok else '❌ FAIL'}  reward={r:.3f}")
            print(f"  {'Structural':12s} [{self._bar(s)}] {s:.2f}")
            print(f"  {'Cost':12s} [{self._bar(c)}] {c:.2f}")
            print(f"  {'Geometry':12s} [{self._bar(g)}] {g:.2f}")
            sf = self.safety_factors[-1]
            if sf > 0:
                print(f"  Safety factor: {sf:.2f}{'  ⚠️  gaming?' if sf > 20 else ''}")
            if self.rewards[-1] == -1.0:
                print(f"  ↳ Code crashed (syntax error or missing `result` variable)")
            print()

        # ── 3D design visualization ──────────────────────────────────────────
        mesh_to_show = None
        if self.passed[-1] if self.passed else False:
            # Try to show the current episode's mesh if it exists
            # mesh_file would need to be stored — see _last_mesh_file
            pass

        if self.best["mesh_file"]:
            self._show_mesh_3d(
                self.best["mesh_file"],
                title=(f"🏆 Best Design — Episode {self.best['episode']} — "
                       f"Score {self.best['reward']:.3f}"),
                color="gold",
            )

        # ── Learning curves ──────────────────────────────────────────────────
        if n >= 5:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Reward history",
                    f"Pass rate (rolling {self.rolling_window})",
                    "Dimension scores (rolling avg)",
                    "Safety factor (passing designs)",
                ),
                vertical_spacing=0.18,
                horizontal_spacing=0.12,
            )

            eps = self.episodes

            # Reward
            fig.add_trace(go.Scatter(
                x=eps, y=self.rewards,
                mode="lines", line=dict(color="#4C9BE8", width=1.5),
                name="reward",
            ), row=1, col=1)
            fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                          annotation_text="pass threshold", row=1, col=1)

            # Rolling pass rate
            w = min(self.rolling_window, n)
            rpr = np.convolve(self.passed, np.ones(w) / w, mode="valid") * 100
            fig.add_trace(go.Scatter(
                x=eps[w - 1:], y=rpr,
                mode="lines", line=dict(color="#2ECC71", width=2),
                name="pass %",
            ), row=1, col=2)
            fig.add_hline(y=self.target_pass_rate * 100,
                          line_dash="dash", line_color="gold",
                          annotation_text=f"target {target_pct}%", row=1, col=2)
            fig.add_hline(y=14, line_dash="dot", line_color="gray",
                          annotation_text="zero-shot", row=1, col=2)

            # Dimension scores rolling avg
            w2 = min(20, n)
            def roll(arr): return np.convolve(arr, np.ones(w2) / w2, mode="valid").tolist()
            x2 = eps[w2 - 1:]
            fig.add_trace(go.Scatter(x=x2, y=roll(self.structural_scores),
                mode="lines", name="structural", line=dict(color="orange")), row=2, col=1)
            fig.add_trace(go.Scatter(x=x2, y=roll(self.cost_scores),
                mode="lines", name="cost", line=dict(color="purple")), row=2, col=1)
            fig.add_trace(go.Scatter(x=x2, y=roll(self.geometry_scores),
                mode="lines", name="geometry", line=dict(color="teal")), row=2, col=1)

            # SF scatter (passing designs only)
            sf_eps  = [e for e, p, s in zip(eps, self.passed, self.safety_factors) if p and s > 0]
            sf_vals = [s for p, s in zip(self.passed, self.safety_factors) if p and s > 0]
            if sf_eps:
                colors = ["red" if s > 20 else "steelblue" for s in sf_vals]
                fig.add_trace(go.Scatter(
                    x=sf_eps, y=sf_vals,
                    mode="markers",
                    marker=dict(color=colors, size=5),
                    name="SF",
                ), row=2, col=2)
                fig.add_hline(y=3.0, line_dash="dash", line_color="green",
                              annotation_text="target SF", row=2, col=2)
                fig.add_hline(y=20, line_dash="dash", line_color="red",
                              annotation_text="gaming?", row=2, col=2)

            fig.update_layout(
                height=520, showlegend=True,
                margin=dict(l=40, r=20, t=60, b=20),
                legend=dict(orientation="h", y=-0.02),
            )
            fig.update_xaxes(title_text="Episode")
            fig.show()

        # ── Best design stats ────────────────────────────────────────────────
        if self.best["episode"] is not None:
            d = self.best["dim_scores"]
            print(f"\n  🏆 Best: ep {self.best['episode']} | "
                  f"reward {self.best['reward']:.3f} | "
                  f"struct {d.get('structural', 0):.2f} | "
                  f"cost {d.get('cost', 0):.2f} | "
                  f"geom {d.get('geometry', 0):.2f}")

        print(f"\n{'━'*68}\n")

    def final_summary(self) -> None:
        """Call after training finishes for the full summary view."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("plotly not installed")
            return

        clear_output(wait=True)
        n = len(self.episodes)
        q = max(1, n // 4)

        initial_pr = np.mean(self.passed[:q])
        final_pr   = np.mean(self.passed[-q:])
        improvement = final_pr - initial_pr

        print(f"\n{'━'*68}")
        print(f"  🎉  TRAINING COMPLETE — {n} episodes")
        print(f"{'━'*68}")
        print(f"  Initial pass rate (first {q}):  {initial_pr:.1%}")
        print(f"  Final pass rate   (last  {q}):  {final_pr:.1%}")
        print(f"  Improvement:                    {improvement:+.1%}")
        print(f"  Best score ever:                {self.best['reward']:.3f}  (ep {self.best['episode']})")
        print(f"  Achievements unlocked ({len(self.unlocked_achievements)}/{len(ACHIEVEMENTS)}):")
        for k in self.unlocked_achievements:
            icon, name, desc = ACHIEVEMENTS[k]
            print(f"    {icon} {name} — {desc}")
        print(f"{'━'*68}\n")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Full reward history",
                "Pass rate evolution",
                "SF distribution (passing designs)",
                "Cost distribution (passing designs)",
            ),
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )
        eps = self.episodes
        w = min(self.rolling_window, n)

        fig.add_trace(go.Scatter(x=eps, y=self.rewards, mode="lines",
            line=dict(color="#4C9BE8", width=1)), row=1, col=1)

        rpr = np.convolve(self.passed, np.ones(w) / w, mode="valid") * 100
        fig.add_trace(go.Scatter(x=eps[w - 1:], y=rpr, mode="lines",
            line=dict(color="#2ECC71", width=2)), row=1, col=2)
        fig.add_hline(y=self.target_pass_rate * 100, line_dash="dash",
                      line_color="gold", row=1, col=2)

        sf_pass = [s for p, s in zip(self.passed, self.safety_factors) if p and s > 0]
        if sf_pass:
            fig.add_trace(go.Histogram(x=sf_pass, marker_color="orange",
                nbinsx=25, name="SF"), row=2, col=1)
            fig.add_vline(x=3.0, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_vline(x=20, line_dash="dash", line_color="red", row=2, col=1)

        cost_pass = [c for p, c in zip(self.passed, self.costs_usd) if p and c > 0]
        if cost_pass:
            fig.add_trace(go.Histogram(x=cost_pass, marker_color="purple",
                nbinsx=20, name="cost"), row=2, col=2)

        fig.update_layout(height=700, showlegend=False,
                          title_text="Training Summary", margin=dict(t=80))
        fig.show()

        if self.best["mesh_file"]:
            self._show_mesh_3d(
                self.best["mesh_file"],
                title=(f"🏆 Champion Design — Episode {self.best['episode']} — "
                       f"Score {self.best['reward']:.3f}"),
                color="gold",
            )

    # ── INTERNAL HELPERS ─────────────────────────────────────────────────────

    def _show_mesh_3d(self, stl_path: str, title: str, color: str = "lightblue") -> None:
        try:
            import trimesh
            import plotly.graph_objects as go
        except ImportError:
            return

        try:
            mesh = trimesh.load(stl_path)
            mesh = mesh.subdivide()          # smoother edges, less blocky triangles

            v, f = mesh.vertices, mesh.faces

            fig = go.Figure()

            # Flat-shaded surface — no specular highlights or reflections
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color,
                opacity=0.25,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0),
                showscale=False,
            ))

            # Wireframe overlay — shows the actual geometry cleanly
            edges = mesh.edges_unique
            xe, ye, ze = [], [], []
            for e in edges:
                xe += [v[e[0], 0], v[e[1], 0], None]
                ye += [v[e[0], 1], v[e[1], 1], None]
                ze += [v[e[0], 2], v[e[1], 2], None]

            fig.add_trace(go.Scatter3d(
                x=xe, y=ye, z=ze,
                mode="lines",
                line=dict(color="steelblue", width=1),
                hoverinfo="skip",
            ))

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                    aspectmode="data",
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
                ),
                width=640, height=480,
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=False,
            )
            fig.show()
        except Exception as e:
            print(f"  [3D view unavailable: {e}]")

    def _rolling_pass_rate(self) -> float:
        if not self.passed:
            return 0.0
        window = self.passed[-self.rolling_window:]
        return float(np.mean(window))

    def _rolling_mean(self, arr: list, n: int | None = None) -> float:
        n = n or self.rolling_window
        if not arr:
            return 0.0
        return float(np.mean(arr[-n:]))

    @staticmethod
    def _bar(score: float, width: int = 16) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled)

    def _unlock(self, key: str) -> bool:
        if key not in self.unlocked_achievements:
            self.unlocked_achievements.add(key)
            icon, name, desc = ACHIEVEMENTS[key]
            print(f"\n  ✨ ACHIEVEMENT UNLOCKED: {icon} {name} — {desc}\n")
            return True
        return False

    def _check_achievements(self, episode: int, dim: dict) -> list[str]:
        newly = []

        def check(key: str, condition: bool):
            if condition and key not in self.unlocked_achievements:
                self._unlock(key)
                newly.append(key)

        p = self.passed
        check("first_pass",       any(p))
        check("three_streak",     self.streak >= 3)
        check("five_streak",      self.streak >= 5)
        check("under_budget",     dim.get("cost", 0) >= 0.95)
        check("structurally_s",   dim.get("structural", 0) >= 0.90)
        check("half_century",     episode >= 50)
        check("century",          episode >= 100)
        check("beat_baseline",    self._rolling_pass_rate() > 0.14)
        check("research_grade",   self._rolling_pass_rate() >= 0.50)

        if dim.get("geometry", 0) == 1.0:
            self._perfect_geom_count += 1
        check("perfect_geometry", self._perfect_geom_count >= 3)

        if self.fail_streak >= 10 and p and p[-1]:
            check("comeback", True)

        return newly
