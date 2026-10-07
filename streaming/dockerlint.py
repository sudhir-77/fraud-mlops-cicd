"""Static Dockerfile checks. The image is not built."""

from __future__ import annotations

from pathlib import Path


def lint_dockerfile_text(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    problems: list[str] = []
    froms = [line for line in lines if line.upper().startswith("FROM ")]
    if len(froms) != 1:
        problems.append("Dockerfile must contain exactly one FROM")
    else:
        image = froms[0].split()[1]
        tag = image.split("@", 1)[0]
        if ":" not in tag or tag.endswith(":latest"):
            problems.append("base image must be pinned and must not use latest")
    if any(line.upper().startswith("ADD ") for line in lines):
        problems.append("use COPY instead of ADD")
    if any("sudo " in line.lower() for line in lines):
        problems.append("do not use sudo")
    secret_tokens = ("PASSWORD=", "SECRET_KEY=", "AWS_SECRET", "API_KEY=")
    for line in lines:
        upper = line.upper()
        if any(token in upper for token in secret_tokens):
            problems.append("possible secret in Dockerfile")
            break
    if not any(line.upper().startswith("WORKDIR ") for line in lines):
        problems.append("WORKDIR is required")
    if not any(
        line.upper().startswith("CMD ") or line.upper().startswith("ENTRYPOINT ") for line in lines
    ):
        problems.append("CMD or ENTRYPOINT is required")
    users = [line for line in lines if line.upper().startswith("USER ")]
    if not users:
        problems.append("USER is required")
    else:
        user = users[-1].split(None, 1)[1].strip()
        if user in {"root", "0"}:
            problems.append("final USER must not be root")
    return problems


def lint_dockerfile(path: Path) -> list[str]:
    return lint_dockerfile_text(Path(path).read_text(encoding="utf-8"))
