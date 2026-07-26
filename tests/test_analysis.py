from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from promptfragc.analyze import (
    build_loco_selection,
    compare_selection_policies,
    corruption_aggregated_friedman_tests,
    friedman_tests,
    interaction_decomposition,
)
from promptfragc.constants import PROMPTS


def synthetic_metrics() -> pd.DataFrame:
    rows = []
    corruptions = ["gaussian_noise", "motion_blur", "jpeg_compression"]
    for corruption_index, corruption in enumerate(corruptions):
        for severity in [1, 3, 5]:
            prompt_probabilities = []
            for prompt_index, (prompt_id, template) in enumerate(PROMPTS):
                accuracy = 0.82 - 0.04 * corruption_index - 0.03 * severity
                accuracy += 0.002 * prompt_index * (corruption_index - 1)
                metrics = {
                    "accuracy": accuracy,
                    "ece": 0.05 + 0.003 * prompt_index + 0.01 * corruption_index,
                    "nll": 0.5 + 0.01 * severity,
                    "brier": 0.2 + 0.005 * severity,
                    "aurc": 0.08 + 0.002 * prompt_index + 0.01 * corruption_index,
                    "risk_at_80": 0.1 + 0.002 * prompt_index,
                }
                prompt_probabilities.append(metrics)
                rows.append(
                    {
                        "model": "ViT-B-32",
                        "pretrained": "openai",
                        "corruption": corruption,
                        "severity": severity,
                        "prompt_id": prompt_id,
                        "prompt_template": template,
                        "method": "single",
                        "n_samples": 1000,
                        **metrics,
                    }
                )
            average = {
                key: float(np.mean([item[key] for item in prompt_probabilities]))
                for key in prompt_probabilities[0]
            }
            rows.append(
                {
                    "model": "ViT-B-32",
                    "pretrained": "openai",
                    "corruption": corruption,
                    "severity": severity,
                    "prompt_id": "ensemble",
                    "prompt_template": "ensemble",
                    "method": "ensemble",
                    "n_samples": 1000,
                    **average,
                }
            )
    for prompt_index, (prompt_id, template) in enumerate(PROMPTS):
        rows.append(
            {
                "model": "ViT-B-32",
                "pretrained": "openai",
                "corruption": "clean",
                "severity": 0,
                "prompt_id": prompt_id,
                "prompt_template": template,
                "method": "single",
                "n_samples": 1000,
                "accuracy": 0.9 - 0.001 * prompt_index,
                "ece": 0.04 + 0.001 * prompt_index,
                "nll": 0.3,
                "brier": 0.1,
                "aurc": 0.03,
                "risk_at_80": 0.02,
            }
        )
    return pd.DataFrame(rows)


class AnalysisTest(unittest.TestCase):
    def test_analysis_tables_are_nonempty(self) -> None:
        frame = synthetic_metrics()
        singles = frame[frame["method"] == "single"]
        self.assertFalse(interaction_decomposition(singles).empty)
        self.assertFalse(friedman_tests(singles).empty)
        conservative = corruption_aggregated_friedman_tests(
            singles,
            bootstrap_samples=100,
            seed=20260723,
        )
        self.assertFalse(conservative.empty)
        self.assertEqual(set(conservative["n_corruptions"]), {3})
        self.assertTrue(
            np.isfinite(
                conservative[
                    ["effect_ci95_low", "effect_ci95_high", "p_value_holm_global"]
                ].to_numpy(dtype=float)
            ).all()
        )
        self.assertTrue(
            (
                conservative["effect_ci95_low"]
                <= conservative["effect_size"]
            ).all()
        )
        self.assertTrue(
            (
                conservative["effect_size"]
                <= conservative["effect_ci95_high"]
            ).all()
        )
        loco = build_loco_selection(frame)
        self.assertEqual(loco["held_out_corruption"].nunique(), 3)
        self.assertIn("stability_loco", set(loco["policy"]))
        policy_tests = compare_selection_policies(
            loco,
            bootstrap_samples=100,
            seed=20260723,
        )
        self.assertIn("p_value_holm_global", policy_tests.columns)
        self.assertTrue(
            policy_tests["p_value_holm_global"].between(0.0, 1.0).all()
        )


if __name__ == "__main__":
    unittest.main()
