"""ephemeral_sandbox.py - Ephemeral sandbox for formalism exploration."""
import os, json, time, random, tempfile, shutil, hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from invariants.gcat_invariants import GCATInvariants, InvariantResult
from invariants.bcat_invariants import BCATInvariants

@dataclass
class ExperimentResult:
    experiment_id: str
    timestamp: str
    seed: str
    parameters: Dict[str, Any]
    gcat_results: Dict[str, InvariantResult]
    bcat_results: Dict[str, InvariantResult]
    aggregate: Dict[str, Any]
    artifacts: Dict[str, Any]

class EphemeralSandbox:
    def __init__(self, seed=None, experiment_name="unnamed", collect_artifacts=True):
        self.seed = seed or hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
        self.experiment_name = experiment_name
        self.collect_artifacts = collect_artifacts
        self.gcat = GCATInvariants()
        self.bcat = BCATInvariants()
        self.results = []
        random.seed(self.seed)
        np.random.seed(int(self.seed[:8], 16) % (2**32))

    def create_isolated_env(self):
        temp_dir = tempfile.mkdtemp(prefix=f"stegverse_sandbox_{self.experiment_name}_")
        os.makedirs(os.path.join(temp_dir, "inputs"))
        os.makedirs(os.path.join(temp_dir, "outputs"))
        os.makedirs(os.path.join(temp_dir, "logs"))
        return temp_dir

    def destroy_env(self, temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_simplex_data(self, dimension=3, edge_range=(0.001, 10.0), collapse_probability=0.1):
        num_edges = dimension * (dimension - 1) // 2
        edges = []
        for _ in range(num_edges):
            edges.append(random.uniform(0.0, 0.0001) if random.random() < collapse_probability else random.uniform(*edge_range))
        triangles = [(edges[i], edges[i+1], edges[i+2]) for i in range(0, len(edges)-2, 3) if i+2 < len(edges)]
        return {"edge_lengths": edges, "triangles": triangles, "dimension": dimension}

    def generate_metric_data(self, dimension=3, positive_definite=True, near_singular=False):
        A = np.random.randn(dimension, dimension)
        G = A @ A.T
        if near_singular:
            eigenvalues, eigenvectors = np.linalg.eigh(G)
            eigenvalues[0] = 1e-12
            G = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return {"metric_tensor": G.tolist()}

    def generate_admissibility_data(self, alpha=None, boundary_proximity="none"):
        if alpha is not None:
            return {"alpha": alpha}
        if boundary_proximity == "danger":
            alpha = random.choice([0.0, 1.0]) + random.uniform(-1e-6, 1e-6)
        elif boundary_proximity == "warning":
            alpha = random.choice([0.0, 1.0]) + random.uniform(0.01, 0.1)
        else:
            alpha = random.uniform(0.1, 0.9)
        return {"alpha": alpha}

    def generate_monotonicity_data(self, length=10, violation_probability=0.0):
        evidence = sorted([random.uniform(0, 1) for _ in range(length)])
        confidence = []
        for i, e in enumerate(evidence):
            base = e * 0.8 + 0.1
            if random.random() < violation_probability and i > 0:
                base = max(0, confidence[-1] - random.uniform(0.1, 0.3))
            confidence.append(base)
        return {"evidence_sequence": evidence, "confidence_sequence": confidence}

    def generate_irreversibility_data(self, num_actions=10, saturation_probability=0.0):
        actions = []
        scores = []
        for i in range(num_actions):
            actions.append({"id": i, "type": random.choice(["read", "write", "delete", "admin"]),
                          "timestamp": time.time_ns(), "context": {"session": random.randint(1, 5)}})
            scores.append(0.99 + random.uniform(0, 0.01) if random.random() < saturation_probability else random.uniform(0.1, 0.8))
        return {"action_history": actions, "irreversibility_scores": scores}

    def run_experiment(self, parameters, custom_data=None):
        temp_dir = self.create_isolated_env()
        experiment_id = f"EXP-{self.experiment_name}-{int(time.time())}"
        try:
            test_data = custom_data or self._generate_test_data(parameters)
            with open(os.path.join(temp_dir, "inputs", "test_data.json"), "w") as f:
                json.dump(test_data, f, indent=2)
            gcat_results = self.gcat.evaluate_all(test_data)
            history = test_data.get("action_history", [])
            bcat_results = self.bcat.evaluate_all(test_data, history)
            aggregate = self._aggregate_results(gcat_results, bcat_results)
            artifacts = {"gcat_metadata": {k: v.metadata for k, v in gcat_results.items()},
                        "bcat_metadata": {k: v.metadata for k, v in bcat_results.items()}} if self.collect_artifacts else {}
            result = ExperimentResult(experiment_id, datetime.now().isoformat(), self.seed,
                                     parameters, gcat_results, bcat_results, aggregate, artifacts)
            self.results.append(result)
            return result
        finally:
            self.destroy_env(temp_dir)

    def _generate_test_data(self, parameters):
        data = {}
        if parameters.get("test_simplex", False):
            data.update(self.generate_simplex_data(parameters.get("dimension", 3), collapse_probability=parameters.get("collapse_prob", 0.1)))
        if parameters.get("test_metric", False):
            data.update(self.generate_metric_data(parameters.get("dimension", 3), parameters.get("positive_definite", True), parameters.get("near_singular", False)))
        if parameters.get("test_admissibility", False):
            data.update(self.generate_admissibility_data(boundary_proximity=parameters.get("boundary_proximity", "none")))
        if parameters.get("test_monotonicity", False):
            data.update(self.generate_monotonicity_data(parameters.get("sequence_length", 10), parameters.get("violation_prob", 0.0)))
        if parameters.get("test_irreversibility", False):
            data.update(self.generate_irreversibility_data(parameters.get("num_actions", 10), parameters.get("saturation_prob", 0.0)))
        return data

    def _aggregate_results(self, gcat, bcat):
        all_results = list(gcat.values()) + list(bcat.values())
        passed = sum(1 for r in all_results if r.passed)
        total = len(all_results)
        return {"total_invariants": total, "total_passed": passed, "total_failed": total - passed,
                "pass_rate": passed / total if total > 0 else 0,
                "mean_confidence": sum(r.confidence for r in all_results) / len(all_results) if all_results else 0,
                "critical_failures": [r.name for r in all_results if not r.passed and r.name.startswith(("I3", "I6", "B2", "B5"))]}

    def run_parameter_sweep(self, parameter_grid, num_runs_per_config=5):
        from itertools import product
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        all_results = []
        for config_values in product(*values):
            config = dict(zip(keys, config_values))
            for _ in range(num_runs_per_config):
                self.seed = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
                random.seed(self.seed)
                np.random.seed(int(self.seed[:8], 16) % (2**32))
                result = self.run_experiment(config)
                all_results.append(result)
        return all_results

    def export_results(self, filepath="formalism_exploration_results.json"):
        data = [{"experiment_id": r.experiment_id, "timestamp": r.timestamp, "seed": r.seed,
                 "parameters": r.parameters, "aggregate": r.aggregate} for r in self.results]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(self.results)} experiments to {filepath}")

    def analyze_for_formalism(self):
        if not self.results:
            return {}
        all_metadata = []
        for r in self.results:
            for inv in list(r.gcat_results.values()) + list(r.bcat_results.values()):
                all_metadata.append({"experiment_id": r.experiment_id, "invariant": inv.name,
                                    "passed": inv.passed, "confidence": inv.confidence, **inv.metadata})
        return {"total_experiments": len(self.results), "pass_rate_by_invariant": self._pass_rate_by_invariant(all_metadata),
                "recommended_changes": self._recommend_changes(all_metadata)}

    def _pass_rate_by_invariant(self, metadata):
        from collections import defaultdict
        counts = defaultdict(lambda: {"passed": 0, "total": 0})
        for m in metadata:
            counts[m["invariant"]]["total"] += 1
            if m["passed"]:
                counts[m["invariant"]]["passed"] += 1
        return {k: v["passed"] / v["total"] for k, v in counts.items()}

    def _recommend_changes(self, metadata):
        pass_rates = self._pass_rate_by_invariant(metadata)
        recs = []
        for inv, rate in pass_rates.items():
            if rate == 1.0:
                recs.append(f"{inv}: Always passes — may be too lenient")
            elif rate == 0.0:
                recs.append(f"{inv}: Always fails — may be too strict")
            elif rate < 0.3:
                recs.append(f"{inv}: Low pass rate ({rate:.1%}) — review threshold")
        return recs
