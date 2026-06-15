from __future__ import annotations

from pathlib import Path

ALLOWED_WORKFLOWS = {
    "core-lite-posture-smoke-test.yml",
}

WORKFLOW_DIR = Path(".github/workflows")
STATUS_PATH = Path("reports/core_lite/latest_status.txt")


def _write_status(actual: list[str], unexpected: list[str], missing: list[str]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not unexpected and not missing else "FAIL"
    STATUS_PATH.write_text(
        "\n".join([
            f"status={status}",
            f"allowed_workflows={','.join(sorted(ALLOWED_WORKFLOWS))}",
            f"actual_workflows={','.join(actual)}",
            f"unexpected_workflows={','.join(unexpected)}",
            f"missing_required_workflows={','.join(missing)}",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if not WORKFLOW_DIR.exists():
        print("FAIL: .github/workflows directory missing")
        return 1

    actual = sorted(p.name for p in WORKFLOW_DIR.iterdir() if p.is_file() and p.suffix in {".yml", ".yaml"})
    unexpected = [name for name in actual if name not in ALLOWED_WORKFLOWS]
    missing = [name for name in sorted(ALLOWED_WORKFLOWS) if name not in actual]

    _write_status(actual, unexpected, missing)

    print("allowed_workflows=", sorted(ALLOWED_WORKFLOWS))
    print("actual_workflows=", actual)

    if unexpected:
        print("FAIL: unexpected active workflows:", unexpected)
        return 1
    if missing:
        print("FAIL: missing required workflows:", missing)
        return 1

    print("PASS: core-lite workflow posture verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
