"""bcat_invariants.py - BCAT boundary checks."""
import math
from typing import Dict, Any, List, Optional
import numpy as np
from invariants.gcat_invariants import InvariantResult

class BCATInvariants:
    def __init__(self, warning_threshold=0.1, danger_threshold=0.01):
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.history = []

    def check_boundary_proximity(self, alpha, context=None):
        min_dist = min(abs(alpha), abs(alpha - 1.0))
        if min_dist < self.danger_threshold:
            status, passed, conf = "DANGER", False, 0.0
        elif min_dist < self.warning_threshold:
            status, passed, conf = "WARNING", True, min_dist / self.warning_threshold
        else:
            status, passed, conf = "SAFE", True, 1.0
        return InvariantResult("B1_BoundaryProximity", passed, conf,
            f"alpha={alpha:.6f}, dist={min_dist:.6f} → {status}", {"alpha": alpha, "status": status})

    def check_metric_degeneracy(self, metric_tensor, context=None):
        G = np.array(metric_tensor)
        try:
            eigenvalues = np.linalg.eigvalsh(G)
            min_eig = float(np.min(eigenvalues))
            max_eig = float(np.max(eigenvalues))
            cond = max_eig / min_eig if min_eig > 1e-12 else float('inf')
        except:
            return InvariantResult("B2_MetricDegeneracy", False, 0.0, "Failed", {})
        passed = cond < 1e12
        return InvariantResult("B2_MetricDegeneracy", passed, 1.0 if passed else 0.0,
            f"cond={cond:.2e}", {"condition_number": cond if cond != float('inf') else None})

    def check_edge_collapse(self, edge_lengths, context=None):
        if not edge_lengths:
            return InvariantResult("B3_EdgeCollapse", False, 0.0, "Empty", {})
        min_edge = min(edge_lengths)
        if min_edge < 1e-7:
            status, passed, conf = "COLLAPSED", False, 0.0
        elif min_edge < 1e-6:
            status, passed, conf = "IMMINENT", True, 0.5
        else:
            status, passed, conf = "SAFE", True, 1.0
        return InvariantResult("B3_EdgeCollapse", passed, conf,
            f"min={min_edge:.6f} → {status}", {"min_edge": min_edge, "status": status})

    def check_confidence_cliff(self, confidence_sequence, context=None):
        if len(confidence_sequence) < 2:
            return InvariantResult("B4_ConfidenceCliff", False, 0.0, "Need >=2", {})
        cliffs = [i for i in range(len(confidence_sequence)-1)
                  if confidence_sequence[i+1] - confidence_sequence[i] < -0.3]
        passed = len(cliffs) < len(confidence_sequence) / 3
        return InvariantResult("B4_ConfidenceCliff", passed, 1.0 - len(cliffs)/len(confidence_sequence),
            f"{len(cliffs)} cliffs", {"cliff_count": len(cliffs)})

    def check_irreversibility_saturation(self, irreversibility_scores, context=None):
        if not irreversibility_scores:
            return InvariantResult("B5_IrreversibilitySaturation", False, 0.0, "Empty", {})
        current_max = max(irreversibility_scores)
        headroom = 1.0 - current_max
        if headroom < 0.01:
            status, passed, conf = "SATURATED", False, 0.0
        elif headroom < 0.1:
            status, passed, conf = "WARNING", True, headroom / 0.1
        else:
            status, passed, conf = "SAFE", True, 1.0
        return InvariantResult("B5_IrreversibilitySaturation", passed, conf,
            f"max={current_max:.6f}, headroom={headroom:.6f} → {status}",
            {"headroom": headroom, "status": status})

    def check_action_replay_resistance(self, action, history, context=None):
        if not history:
            return InvariantResult("B6_ActionReplayResistance", True, 1.0, "No history", {})
        similar = [h for h in history if h.get("action") == action.get("action")]
        exact = [h for h in similar if h.get("context") == action.get("context")]
        if exact:
            status, passed, conf = "REPLAY_DETECTED", False, 0.0
        elif similar:
            status, passed, conf = "CONTEXT_DEPENDENT", True, 0.8
        else:
            status, passed, conf = "UNIQUE", True, 1.0
        return InvariantResult("B6_ActionReplayResistance", passed, conf,
            f"{len(similar)} similar, {len(exact)} exact → {status}", {"status": status})

    def evaluate_all(self, test_data, history=None):
        results = {}
        if "alpha" in test_data:
            results["B1"] = self.check_boundary_proximity(test_data["alpha"])
        if "metric_tensor" in test_data:
            results["B2"] = self.check_metric_degeneracy(test_data["metric_tensor"])
        if "edge_lengths" in test_data:
            results["B3"] = self.check_edge_collapse(test_data["edge_lengths"])
        if "confidence_sequence" in test_data:
            results["B4"] = self.check_confidence_cliff(test_data["confidence_sequence"])
        if "irreversibility_scores" in test_data:
            results["B5"] = self.check_irreversibility_saturation(test_data["irreversibility_scores"])
        if "action" in test_data and history:
            results["B6"] = self.check_action_replay_resistance(test_data["action"], history)
        return results
