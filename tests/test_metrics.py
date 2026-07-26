from __future__ import annotations

import unittest

import numpy as np

from promptfragc.metrics import (
    area_under_risk_coverage,
    classification_metrics,
    expected_calibration_error,
)


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        probabilities = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.95, 0.05],
                [0.1, 0.9],
            ]
        )
        targets = np.array([0, 1, 0, 1])
        result = classification_metrics(probabilities, targets, n_bins=10)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["aurc"], 0.0)
        self.assertEqual(result["risk_at_80"], 0.0)
        self.assertGreater(result["nll"], 0.0)

    def test_risk_sorting(self) -> None:
        confidence = np.array([0.95, 0.90, 0.60, 0.55])
        correct = np.array([1, 1, 0, 0], dtype=bool)
        aurc, risk_80 = area_under_risk_coverage(confidence, correct)
        self.assertGreater(aurc, 0.0)
        self.assertAlmostEqual(risk_80, 0.5)

    def test_ece_known_value(self) -> None:
        confidence = np.array([0.8, 0.8])
        correct = np.array([1, 0], dtype=bool)
        self.assertAlmostEqual(
            expected_calibration_error(confidence, correct, n_bins=10),
            0.3,
        )

    def test_rejects_non_probabilities(self) -> None:
        probabilities = np.array([[0.8, 0.8]])
        with self.assertRaises(ValueError):
            classification_metrics(probabilities, np.array([0]))


if __name__ == "__main__":
    unittest.main()

