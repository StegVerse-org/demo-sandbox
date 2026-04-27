"""experiment_suite.py - Pre-built formalism experiment suites."""
import json, numpy as np
from typing import Dict, Any, List
from sandbox.ephemeral_sandbox import EphemeralSandbox, ExperimentResult

class ExperimentSuites:
    def __init__(self, base_seed="formalism-exploration"):
        self.base_seed = base_seed
        self.all_results = []

    def suite_edge_collapse_convergence(self, num_steps=20):
        print("\n[SUITE 1] Edge Collapse → ALLOW Convergence")
        print("-" * 40)
        results = []
        sandbox = EphemeralSandbox(seed=f"{self.base_seed}-collapse", experiment_name="collapse")
        for step in range(num_steps):
            scale = 10.0 * (0.5 ** step)
            result = sandbox.run_experiment({"test_simplex": True, "dimension": 3, "collapse_prob": 0.0,
                                            "test_admissibility": True, "alpha": 0.5},
                                           custom_data={"edge_lengths": [scale, scale * 1.1, scale * 0.9],
                                                       "triangles": [(scale, scale * 1.1, scale * 0.9)], "alpha": 0.5})
            results.append(result)
            gcat_pass = result.aggregate.get("gcat_pass_rate", 0)
            bcat_pass = result.aggregate.get("bcat_pass_rate", 0)
            print(f"  Step {step:2d}: scale={scale:.6f} | GCAT={gcat_pass:.1%} | BCAT={bcat_pass:.1%}")
            if scale < 1e-9:
                print(f"  → Collapse threshold reached at step {step}")
                break
        self.all_results.extend(results)
        return results

    def suite_alpha_threshold(self, num_samples=100):
        print("\n[SUITE 2] Alpha Boundary Threshold Optimization")
        print("-" * 40)
        results = []
        sandbox = EphemeralSandbox(seed=f"{self.base_seed}-alpha", experiment_name="alpha")
        alphas = np.linspace(0, 1, num_samples)
        for alpha in alphas:
            result = sandbox.run_experiment({"test_admissibility": True, "alpha": float(alpha), "test_simplex": True, "dimension": 3},
                                           custom_data={"alpha": float(alpha), "edge_lengths": [1.0, 1.0, 1.0], "triangles": [(1.0, 1.0, 1.0)]})
            results.append(result)
        failures = [(alphas[i], r.bcat_results.get("B1").metadata.get("min_dist", 0)) for i, r in enumerate(results) if r.bcat_results.get("B1") and not r.bcat_results["B1"].passed]
        if failures:
            print(f"  Failures start at α ≈ {min(f[0] for f in failures):.4f}")
        self.all_results.extend(results)
        return results

    def suite_monotonicity_contradiction(self, violation_rates=None):
        if violation_rates is None:
            violation_rates = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        print("\n[SUITE 3] Confidence Monotonicity Under Contradiction")
        print("-" * 40)
        results = []
        sandbox = EphemeralSandbox(seed=f"{self.base_seed}-mono", experiment_name="monotonicity")
        for rate in violation_rates:
            result = sandbox.run_experiment({"test_monotonicity": True, "sequence_length": 50, "violation_prob": rate})
            i5 = result.gcat_results.get("I5")
            b4 = result.bcat_results.get("B4")
            print(f"  Violation rate {rate:.1%}: I5_pass={i5.passed if i5 else False} | I5_conf={i5.confidence if i5 else 0:.3f}")
            results.append(result)
        self.all_results.extend(results)
        return results

    def suite_metric_degeneracy(self, dimensions=None):
        if dimensions is None:
            dimensions = [2, 3, 4, 5, 10]
        print("\n[SUITE 4] Metric Degeneracy Trigger Points")
        print("-" * 40)
        results = []
        sandbox = EphemeralSandbox(seed=f"{self.base_seed}-metric", experiment_name="metric")
        for dim in dimensions:
            for spread in [1, 10, 100, 1000, 10000, 100000, 1000000]:
                eigenvalues = [1.0] + [1.0 / spread] * (dim - 1)
                D = np.diag(eigenvalues)
                Q = np.linalg.qr(np.random.randn(dim, dim))[0]
                G = Q @ D @ Q.T
                result = sandbox.run_experiment({"test_metric": True, "dimension": dim},
                                               custom_data={"metric_tensor": G.tolist()})
                i3 = result.gcat_results.get("I3")
                b2 = result.bcat_results.get("B2")
                cond = i3.metadata.get("condition_number", 0) if i3 else 0
                print(f"  Dim={dim:2d} spread={spread:8d} cond={cond:.2e} I3={i3.passed if i3 else False} B2={b2.metadata.get('status', 'N/A') if b2 else 'N/A'}")
                results.append(result)
        self.all_results.extend(results)
        return results

    def suite_rigel_derivation(self, num_samples=500):
        print("\n[SUITE 5] Empirical Rigel Metric Derivation")
        print("-" * 40)
        results = []
        sandbox = EphemeralSandbox(seed=f"{self.base_seed}-rigel", experiment_name="rigel")
        data_points = []
        for i in range(num_samples):
            dim = random.choice([2, 3, 4, 5])
            edges = [random.uniform(0.1, 10.0) for _ in range(dim * (dim - 1) // 2)]
            result = sandbox.run_experiment({"test_simplex": True, "dimension": dim},
                                           custom_data={"edge_lengths": edges,
                                                       "triangles": [(edges[j], edges[j+1], edges[j+2]) for j in range(0, len(edges)-2, 3) if j+2 < len(edges)]})
            mean_conf = result.aggregate.get("mean_confidence", 0)
            data_points.append({"edges": edges, "confidence": mean_conf, "passed": result.aggregate.get("pass_rate", 0) > 0.5})
            results.append(result)
        bins = {"high": [], "mid": [], "low": []}
        for dp in data_points:
            if dp["confidence"] > 0.7: bins["high"].append(dp["edges"])
            elif dp["confidence"] > 0.3: bins["mid"].append(dp["edges"])
            else: bins["low"].append(dp["edges"])
        print(f"  Data: {len(data_points)} points | High: {len(bins['high'])} | Mid: {len(bins['mid'])} | Low: {len(bins['low'])}")
        self.all_results.extend(results)
        return results

    def run_all(self):
        print("=" * 50)
        print("STEGVERSE FORMALISM EXPLORATION — ALL SUITES")
        print("=" * 50)
        self.suite_edge_collapse_convergence()
        self.suite_alpha_threshold()
        self.suite_monotonicity_contradiction()
        self.suite_metric_degeneracy()
        self.suite_rigel_derivation()
        analysis = self._cross_suite_analysis()
        print(f"\nTotal experiments: {len(self.all_results)}")
        print(f"Overall pass rate: {analysis['overall_pass_rate']:.1%}")
        self._export_all()

    def _cross_suite_analysis(self):
        total_checks = 0
        total_passed = 0
        for r in self.all_results:
            agg = r.aggregate
            total_checks += agg.get("total_invariants", 0)
            total_passed += agg.get("total_passed", 0)
        return {"overall_pass_rate": total_passed / total_checks if total_checks else 0, "total_experiments": len(self.all_results)}

    def _export_all(self):
        data = [{"experiment_id": r.experiment_id, "timestamp": r.timestamp, "seed": r.seed,
                 "parameters": r.parameters, "aggregate": r.aggregate} for r in self.all_results]
        with open("formalism_exploration_results.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nExported to formalism_exploration_results.json")

    def export_results(self, filepath="formalism_exploration_results.json"):
        self._export_all()

if __name__ == "__main__":
    import random
    suites = ExperimentSuites()
    suites.run_all()
