"""Abstract simulator interface. All domain simulators implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SimulationResult:
    success: bool
    mesh_file: str | None = None          # Path to generated geometry (CAD domains)
    outputs: dict = field(default_factory=dict)   # Domain-specific numeric outputs
    error: str | None = None


class BaseSimulator(ABC):
    """
    Executes agent-generated code and produces a SimulationResult.

    The agent's action is always a Python code string. The simulator:
      1. Executes it in a sandboxed namespace
      2. Extracts the design artifact (STL, circuit netlist, etc.)
      3. Runs domain-specific analysis (FEA, SPICE, energy sim)
      4. Returns structured SimulationResult
    """

    # Subclasses declare what names the agent's code must define
    REQUIRED_VARIABLES: list[str] = []

    @abstractmethod
    def execute(self, code: str, task_spec: dict) -> SimulationResult:
        """Execute agent code and return simulation results."""

    def _safe_exec(self, code: str, allowed_imports: list[str] | None = None) -> dict:
        """
        Execute agent code and return the populated namespace.

        Pre-injects allowed_imports as top-level names so agents can use them
        without needing an import statement. Full builtins are available —
        this is NOT a security sandbox. For untrusted code, run in a subprocess.

        Raises RuntimeError on execution failure or missing required variables.
        """
        import importlib

        # Full builtins required: CadQuery (and most domain libs) use import
        # statements internally that need __import__ to resolve.
        namespace: dict = {"__builtins__": __builtins__}

        if allowed_imports:
            for mod_name in allowed_imports:
                top = mod_name.split(".")[0]
                namespace[top] = importlib.import_module(mod_name)

        try:
            exec(compile(code, "<agent>", "exec"), namespace)
        except Exception as exc:
            raise RuntimeError(f"Agent code failed: {exc}") from exc

        for var in self.REQUIRED_VARIABLES:
            if var not in namespace:
                raise RuntimeError(f"Agent code must define variable '{var}'")

        return namespace
