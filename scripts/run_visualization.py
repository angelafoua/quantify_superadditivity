"""Generate publication-quality figures from experiment results.

Usage:
    python scripts/run_visualization.py --results_dir outputs/core_factorial
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.utils.io import ensure_dir, load_json
from superadditivity.visualization.figure_style import set_publication_style

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.output_dir or str(results_dir / "figures"))
    ensure_dir(fig_dir)

    set_publication_style()

    agg = ResultAggregator(str(results_dir))
    df = agg.load_summaries()

    if df.empty:
        logger.error("No data found in %s", results_dir)
        return

    if "data_regime" in df.columns and "network_regime" in df.columns:
        from superadditivity.visualization.interaction_plots import plot_factorial_grid
        fig = plot_factorial_grid(
            df,
            save_path=str(fig_dir / "factorial_grid.pdf"),
        )
        logger.info("Generated factorial grid")
        import matplotlib.pyplot as plt
        plt.close(fig)

    trajectories = agg.load_trajectories(metric="cka", layer="layer4")
    if trajectories:
        from superadditivity.visualization.drift_trajectories import plot_drift_trajectories
        import numpy as np

        traj_by_condition = {}
        for run_name, arr in trajectories.items():
            condition = "_".join(run_name.split("_")[:2]) if "_" in run_name else run_name
            if condition not in traj_by_condition:
                traj_by_condition[condition] = []
            traj_by_condition[condition].append(arr)

        plot_data = {
            k: np.array(v) for k, v in traj_by_condition.items()
        }
        fig = plot_drift_trajectories(
            plot_data,
            save_path=str(fig_dir / "drift_trajectories.pdf"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        logger.info("Generated drift trajectories")

    surface_path = results_dir / "analysis" / "interaction_surface.json"
    if surface_path.exists():
        from superadditivity.visualization.interaction_surface import plot_interaction_surface
        import numpy as np

        surface_data = load_json(surface_path)
        fig = plot_interaction_surface(
            np.array(surface_data["alpha_values"]),
            np.array(surface_data["beta_values"]),
            np.array(surface_data["I_values"]),
            save_path=str(fig_dir / "interaction_surface.pdf"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        logger.info("Generated interaction surface")

    logger.info("All figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
