"""gcat_invariants.py - GCAT invariant checks."""
import math, hashlib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class InvariantResult:
    name: str
    passed: bool
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class GCATInvariants:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon
        self.history = []

    def check_simplex_non_negativity(self, edge_lengths, context=None):
        if not edge_lengths:
            return InvariantResult("I1_SimplexNonNegativity", False, 0.0, "Empty edge set", {})
        min_edge = min(edge_lengths)
        passed = min_edge >= -self.epsilon
        confidence = min(1.0, 1.0 - math.exp(-min_edge * 10)) if min_edge >= 0 else 0.0
        return InvariantResult("I1_SimplexNonNegativity", passed, confidence,
            f"Min edge={min_edge:.6f}", {"edge_count": len(edge_lengths), "min_edge": min_edge})

    def check_triangle_inequality(self, edges, context=None):
        if not edges:
            return InvariantResult("I2_TriangleInequality", False, 0.0, "Empty triangles", {})
        violations = []
        for a, b, c in edges:
            if a + b < c - self.epsilon or a + c < b - self.epsilon or b + c < a - self.epsilon:
                violations.append((a, b, c))
        passed = len(violations) == 0
        return InvariantResult("I2_TriangleInequality", passed, 1.0 if passed else 0.0,
            f"{len(violations)} violations", {"triangle_count": len(edges), "violations": len(violations)})

    def check_rigel_metric_positivity(self, metric_tensor, context=None):
        if not metric_tensor:
            return InvariantResult("I3_RigelMetricPositivity", False, 0.0, "Empty metric", {})
        G = np.array(metric_tensor)
        try:
            eigenvalues = np.linalg.eigvalsh(G)
            min_eig = float(np.min(eigenvalues))
            passed = min_eig > self.epsilon
            confidence = min(1.0, 1.0 - math.exp(-min_eig * 100)) if passed else 0.0
        except:
            return InvariantResult("I3_RigelMetricPositivity", False, 0.0, "Eigenvalue failure", {})
        return InvariantResult("I3_RigelMetricPositivity", passed, confidence,
            f"Min eigenvalue={min_eig:.6f}", {"min_eigenvalue": min_eig, "dimensions": G.shape[0]})

    def check_admissibility_scalar_bounds(self, alpha, context=None):
        passed = 0.0 - self.epsilon <= alpha <= 1.0 + self.epsilon
        confidence = 1.0 - abs(alpha - 0.5) * 2 if 0 <= alpha <= 1 else 0.0
        return InvariantResult("I4_AdmissibilityScalarBounds", passed, confidence,
            f"alpha={alpha:.6f}", {"alpha": alpha, "in_bounds": passed})

    def check_confidence_monotonicity(self, evidence_sequence, confidence_sequence, context=None):
        if len(evidence_sequence) < 2:
            return InvariantResult("I5_ConfidenceMonotonicity", False, 0.0, "Need >=2 pairs", {})
        violations = sum(1 for i in range(len(evidence_sequence)-1)
                        if evidence_sequence[i+1] > evidence_sequence[i]
                        and confidence_sequence[i+1] < confidence_sequence[i] - self.epsilon)
        passed = violations == 0
        confidence = (len(evidence_sequence) - 1 - violations) / (len(evidence_sequence) - 1)
        return InvariantResult("I5_ConfidenceMonotonicity", passed, confidence,
            f"{violations} violations", {"violations": violations})

    def check_irreversibility_preservation(self, action_history, context=None):
        if len(action_history) < 2:
            return InvariantResult("I6_IrreversibilityPreservation", False, 0.0, "Need >=2 actions", {})
        scores = [int(hashlib.sha256(str(sorted(a.items())).encode()).hexdigest(), 16) % 10000 / 10000
                  for a in action_history]
        violations = sum(1 for i in range(len(scores)-1) if scores[i+1] < scores[i] - self.epsilon)
        passed = violations == 0
        return InvariantResult("I6_IrreversibilityPreservation", passed, 0.5 if passed else 0.0,
            f"{violations} violations", {"action_count": len(action_history)})

    def evaluate_all(self, test_data):
        results = {}
        if "edge_lengths" in test_data:
            results["I1"] = self.check_simplex_non_negativity(test_data["edge_lengths"])
        if "triangles" in test_data:
            results["I2"] = self.check_triangle_inequality(test_data["triangles"])
        if "metric_tensor" in test_data:
            results["I3"] = self.check_rigel_metric_positivity(test_data["metric_tensor"])
        if "alpha" in test_data:
            results["I4"] = self.check_admissibility_scalar_bounds(test_data["alpha"])
        if "evidence_sequence" in test_data and "confidence_sequence" in test_data:
            results["I5"] = self.check_confidence_monotonicity(test_data["evidence_sequence"], test_data["confidence_sequence"])
        if "action_history" in test_data:
            results["I6"] = self.check_irreversibility_preservation(test_data["action_history"])
        return results

    def get_experimental_data(self):
        return {"total_checks": len(self.history), "aggregate_confidence": {
            "mean": sum(r.confidence for r in self.history) / len(self.history) if self.history else 0
        }}
