"""
Find the minimum of f(x) + lambda * norm1(x), where x is the FIFO depth config and f(x) is the latency of the simulation.
"""

from typing import Literal
import optuna

from sweep_fifo_configs import process_single_config


# Mode: "global" for full search, "local" for search around a known configuration
MODE: Literal["global", "local"] = "local"

# 14 reader FIFO depths, 4 writer FIFO depths
STRUCTURE = [14, 4]
MIN_DEPTH = 1
MAX_DEPTH = 10
N_TRIALS = 50

# Local search parameters (only used when MODE = "local")
STARTING_CONFIG = [9,4,7,2,7,4,1,3,1,6,7,4,5,6 ,    6,6,1,2]  # 14 readers, 4 writers
SEARCH_RADIUS = 2

PROGRAM = "main" if MODE == "global" else "main-full"
LAMBDA = 10 if MODE == "global" else 1


def f(x: tuple[list[int], list[int]]):
    """Wrap the simulation call. x is the FIFO depth config."""
    try:
        return process_single_config(*x, PROGRAM)
    except Exception as e:
        print(f"  ERROR in f(): {e}", flush=True)
        # Return a large penalty value for failed simulations to avoid them being selected
        return 999_999


def flat_to_structured(flat_input: list[int]) -> tuple[list[int], list[int]]:
    """Convert a flat list to a structured tuple."""
    return flat_input[: STRUCTURE[0]], flat_input[STRUCTURE[0] :]


def objective(trial):
    """
    Optuna allows us to write standard Python code to define
    our own complex objective.
    """

    # A. Ask the optimizer to suggest 18 integers and define the search space
    total_vars = sum(STRUCTURE)
    flat_input = []

    for i in range(total_vars):
        if MODE == "local":
            # Local search: define range around starting value
            start_val = STARTING_CONFIG[i]
            low = max(MIN_DEPTH, start_val - SEARCH_RADIUS)
            high = min(MAX_DEPTH, start_val + SEARCH_RADIUS)
        elif MODE == "global":
            # Global search: full range
            low = MIN_DEPTH
            high = MAX_DEPTH
        else:
            raise ValueError(f"Invalid mode: {MODE}")

        # Suggest an integer for dimension i
        val = trial.suggest_int(f"x_{i}", low, high)
        flat_input.append(val)

    # C. Calculate the "Grey Box" Penalty (L1 Norm). We do this here because it's instant and free.
    l1_norm = sum(abs(i) for i in flat_input)

    # D. Evaluate the expensive function
    x = flat_to_structured(flat_input)
    function_val = f(x)

    # E. Return the combined Scalarized Objective
    # Minimize: f(x) + lambda * norm
    return function_val + (LAMBDA * l1_norm)


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"Starting {MODE} optimization with {N_TRIALS} trials")
    if MODE == "local":
        print(f"Starting configuration: {flat_to_structured(STARTING_CONFIG)} ± search radius {SEARCH_RADIUS}")

    study.optimize(objective, n_trials=N_TRIALS)

    print(f"Best Combined Score: {study.best_value}")
    print(f"Best Parameters (Flat): {study.best_params}")
    print(f"Best Input Structure: {flat_to_structured(list(study.best_params.values()))}")
