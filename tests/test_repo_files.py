"""Repository-level checks: licence, CI, launch copy, no placeholders, no model identifiers."""
from __future__ import annotations

import json
import re

import pytest

LAUNCH = ("LINKEDIN.md", "X_THREAD.md", "DISCORD_SHORT.md", "FAQ.md")
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
             "outputs"}


def repo_text_files(root):
    for path in root.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(root).parts):
            if path.suffix in {".py", ".md", ".json", ".toml", ".yml", ".mmd", ".sh", ".txt", ""}:
                if path.name != "SUMMARY.json" and "data/raw" not in path.as_posix():
                    yield path


def test_licence_is_mit_for_the_author(root):
    text = (root / "LICENSE").read_text()
    assert text.startswith("MIT License") and "Copyright (c) 2026 Ammon Jackson" in text


def test_pyproject_declares_the_console_script_and_light_core(root):
    text = (root / "pyproject.toml").read_text()
    assert 'groundrl = "groundrl.cli:main"' in text and 'name = "groundrl"' in text
    core = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "numpy" in core and not re.search(r"torch|transformers|trl|peft|datasets|vllm", core)
    assert 'requires-python = ">=3.10"' in text


def test_ci_runs_lint_and_coverage_on_every_supported_python_without_gpu_dependencies(root):
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f'"{version}"' in ci
    assert "ruff check ." in ci and "pytest -q --cov" in ci
    assert ".[train" not in ci and "pip install torch" not in ci


def test_makefile_and_dockerfile_exist_and_stay_cpu_only(root):
    make = (root / "Makefile").read_text()
    for target in ("test:", "lint:", "cov:", "audit:", "dry-run:", "diagrams:"):
        assert target in make
    docker = (root / "Dockerfile").read_text()
    assert '".[dev]"' in docker and "train" not in docker.split("RUN", 1)[1].split("\n", 1)[0]


def test_render_script_is_the_shared_one_and_diagrams_are_rendered(root):
    script = (root / "scripts" / "render_diagrams.sh").read_text()
    assert "mermaid-cli" in script and "docs/diagrams/*.mmd" in script
    for name in ("usage", "troubleshooting"):
        svg = (root / "docs" / "diagrams" / f"{name}.svg").read_text()
        assert svg.lstrip().startswith("<svg") and "viewBox" in svg


@pytest.mark.parametrize("name", LAUNCH)
def test_launch_copy_never_claims_training_results(root, name):
    text = (root / "docs" / "launch" / name).read_text()
    low = text.lower()
    assert "not yet run" in low or "have not run" in low or "no model" in low or "not run" in low
    banned_phrases = ("state-of-the-art", "outperform", "we trained", "our trained",
                      "improves accuracy", "improved accuracy", "sota", "beats gpt", "% better",
                      "achieves")
    for banned in banned_phrases:
        assert banned not in low, banned
    assert not re.search(r"trained (model|policy)[^.]{0,40}(reach|achiev|score)d?", low)


def test_x_thread_posts_fit_in_280_characters(root):
    text = (root / "docs" / "launch" / "X_THREAD.md").read_text()
    posts = re.split(r"^\d+/\n", text, flags=re.MULTILINE)[1:]
    assert len(posts) >= 6
    for post in posts:
        flat = " ".join(post.split())
        assert len(flat) <= 280, (len(flat), flat[:60])


def test_launch_numbers_match_the_result_files(root):
    audit = json.loads((root / "results" / "audit_test.json").read_text())
    base = json.loads((root / "results" / "baseline_test.json").read_text())
    ext = base["policies"]["extractive"]["metrics"]["mean_reward"]["v2"]["mean"]
    top = max(p["scores"]["v2"]["mean"] for p in audit["policies"].values()
              if p["category"] != "reference")
    gold_free = len(audit["exploitable"]["v2-gold-free"])
    for name in LAUNCH:
        text = (root / "docs" / "launch" / name).read_text()
        if "0.61" in text:
            assert round(ext, 2) == 0.61
        if "0.44" in text:
            assert round(top, 2) == 0.44
        if "7 policies" in text or "7 of the policies" in text or "7 hacks" in text:
            assert gold_free == 7
    summary = json.loads((root / "data" / "tasks" / "SUMMARY.json").read_text())
    assert str(summary["total"]) in (root / "docs" / "launch" / "FAQ.md").read_text()


def test_no_todos_or_placeholder_code_anywhere(root):
    for path in repo_text_files(root):
        if path.suffix in {".py", ".sh", ".toml", ".yml"} and path.name != "test_repo_files.py":
            text = path.read_text(errors="ignore")
            assert not re.search(r"\b(TODO|FIXME|XXX)\b", text), path
            assert "NotImplementedError" not in text, path


def test_no_ai_model_identifiers_outside_base_model_names(root):
    banned = [b"claude", b"anthropic", b"sonnet", b"opus", b"openai", b"chatgpt", b"gemini"]
    for path in repo_text_files(root):
        if path.suffix == ".svg" or path.name == "test_repo_files.py":
            continue
        low = path.read_bytes().lower()
        for word in banned:
            assert word not in low, (path, word)
    for path in (root / "train" / "configs").glob("*.json"):
        name = json.loads(path.read_text())["model"]["name_or_path"]
        assert name.startswith("Qwen/Qwen3-")


def test_gitignore_keeps_training_outputs_out_of_git(root):
    text = (root / ".gitignore").read_text()
    for entry in ("outputs/", "*.safetensors", "__pycache__/"):
        assert entry in text
