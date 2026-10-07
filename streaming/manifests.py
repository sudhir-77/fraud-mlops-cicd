"""Validate Kubernetes YAML against the v1.29 schema. No cluster is contacted."""

from __future__ import annotations

from pathlib import Path

import kubernetes_validate
import yaml

K8S_VERSION = "1.29"
REQUIRED_KINDS = ("Deployment", "Service", "HorizontalPodAutoscaler")


def validate_document(document: dict, version: str = K8S_VERSION) -> tuple[bool, str]:
    kind = str(document.get("kind", "")) if isinstance(document, dict) else ""
    try:
        kubernetes_validate.validate(document, version, strict=True)
    except Exception as exc:
        print(f"manifest {kind or 'unknown'} invalid: {exc}", flush=True)
        return False, kind
    return True, kind


def validate_manifest_dir(directory: Path, version: str = K8S_VERSION) -> dict[str, dict]:
    directory = Path(directory)
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise RuntimeError(f"no manifests in {directory}")
    report: dict[str, dict] = {}
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            report[f"{directory.name}/{path.name}"] = {"valid": False, "kind": ""}
            continue
        valid, kind = validate_document(document, version)
        report[f"{directory.name}/{path.name}"] = {"valid": bool(valid), "kind": kind}
    return report


def manifests_ok(report: dict[str, dict]) -> bool:
    kinds = {str(item.get("kind", "")) for item in report.values()}
    if not set(REQUIRED_KINDS).issubset(kinds):
        return False
    return all(bool(item.get("valid")) for item in report.values())
