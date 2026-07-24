"""Calculate k-projection coverage for the CARLA test scenarios.

The coverage factors are weather, lighting, and town.  Every labelled test
split represents one scenario and contains 3,600 frames.  A k-projection is
covered when a test scenario contains that combination of k factor values.

Run from the project root:
    python scripts/evaluate_odd_coverage.py
"""

import itertools
from pathlib import Path
import warnings

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings(
    "ignore",
    message="pyarrow.feather.read_table is deprecated",
    category=FutureWarning,
)
TEST_SPLITS = {
    "test": "source_town",
    "test-fog": "source_town",
    "test-night": "source_town",
    "test-town-01": "town_01",
}
FACTORS = ("weather", "lighting", "town")


def classify_weather(weather):
    """Map CARLA weather metadata to the three observed test categories."""
    row = weather.iloc[0]
    if row["fog_density"] >= 50:
        return "fog"
    if row["precipitation"] > 0 or row["wetness"] > 0:
        return "rain"
    return "dry"


def classify_lighting(weather):
    """Classify a scenario as day or night from CARLA's sun elevation."""
    return "night" if weather.iloc[0]["sun_altitude_angle"] < 0 else "day"


def load_scenarios(data_dir):
    """Read one weather record per labelled test scenario."""
    scenarios = []
    for split_name, town in TEST_SPLITS.items():
        split_dir = data_dir / split_name
        weather = pd.read_feather(split_dir / "weather.feather")
        labels = pd.read_csv(split_dir / "labels.csv")
        scenarios.append({
            "split": split_name,
            "frames": len(labels),
            "weather": classify_weather(weather),
            "lighting": classify_lighting(weather),
            "town": town,
        })
    return pd.DataFrame(scenarios)


def calculate_coverage(scenarios):
    """Return coverage across all one-, two-, and three-factor projections."""
    levels = {
        factor: sorted(scenarios[factor].unique())
        for factor in FACTORS
    }
    results = []
    for k in range(1, len(FACTORS) + 1):
        factor_sets = list(itertools.combinations(FACTORS, k))
        possible = sum(
            len(list(itertools.product(*(levels[factor] for factor in factor_set))))
            for factor_set in factor_sets
        )
        observed = sum(
            len(scenarios.loc[:, factor_set].drop_duplicates())
            for factor_set in factor_sets
        )
        results.append({
            "k": k,
            "factor_projections": "; ".join(" x ".join(group) for group in factor_sets),
            "observed_combinations": observed,
            "possible_combinations": possible,
            "coverage_percent": 100 * observed / possible,
        })
    return pd.DataFrame(results)


def main():
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "outputs" / "odd"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(data_dir)
    coverage = calculate_coverage(scenarios)
    scenarios.to_csv(output_dir / "odd_test_scenarios.csv", index=False)
    coverage.to_csv(output_dir / "odd_k_projection_coverage.csv", index=False)

    print("Observed test scenarios:")
    print(scenarios.to_string(index=False))
    print("\nk-projection coverage:")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
