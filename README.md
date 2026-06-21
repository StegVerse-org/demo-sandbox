# StegVerse Demo Sandbox

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Release: v1.0.0

Sandbox environment for StegVerse governance experiments, invariant validation, and ephemeral testing.

This repository is a public demo sandbox. It is not the authority-bearing trust kernel, admission layer, or production sandbox route. Bounded adversarial and entity-specific cases should route through `StegGhost/entity-sandbox-runner` after SDK intake.

---

## What it does

- GCAT/BCAT invariant demonstrations;
- ephemeral sandbox fixtures;
- reproducible governance experiments;
- smoke tests for controlled safety gates.

---

## Structure

```text
demo-sandbox/
├── experiments/        # Reproducible experiment definitions
├── invariants/         # GCAT/BCAT invariant demonstrations
├── sandbox/            # Ephemeral sandbox runtime fixtures
└── README.md
```

---

## Install

```bash
pip install stegverse-demo-sandbox
```

---

## Quick start

```python
from sandbox import EphemeralSandbox
from invariants import gcat_invariants, bcat_invariants

sandbox = EphemeralSandbox()

assert gcat_invariants.legitimacy_surplus() >= 0
assert bcat_invariants.observability_score() >= 0.6
```

---

## Integration

| System | Role |
|---|---|
| `StegVerse-org/StegVerse-SDK` | SDK intake and manifest-bound submission. |
| `StegVerse-org/demo_ingest_engine` | Experiment ingestion and route orchestration. |
| `StegGhost/entity-sandbox-runner` | Bounded sandbox route for adversarial/entity tests. |
| Trust Kernel | Private authority-bearing governance kernel. |
| StegVerse Admission | Private admission / threshold layer. |

---

## Boundary rule

Demo sandbox results are experimental and bounded. They do not create execution authority, external validation, endorsement, compatibility recognition, provenance recognition, collaboration, or public attribution.

---

## Links

- Repository: https://github.com/StegVerse-org/demo-sandbox
- Issues: https://github.com/StegVerse-org/demo-sandbox/issues
