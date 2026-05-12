from __future__ import annotations

import ast
from pathlib import Path

from tomlkit import dumps, parse, table


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "crystall" / "core.py"
README_PATH = ROOT / "README.md"
PYPROJECT_PATH = ROOT / "pyproject.toml"
CITATION_PATH = ROOT / "citation.cff"
DOCS_SOURCE = ROOT / "docs" / "_source"

README_SOURCES = [
    DOCS_SOURCE / "summary.md",
    DOCS_SOURCE / "installation.md",
    DOCS_SOURCE / "features.md",
    DOCS_SOURCE / "contributing.md",
    DOCS_SOURCE / "citing.md",
]

METADATA_KEYS = {
    "__name__",
    "__version__",
    "__authors__",
    "__doc__",
    "__rep__",
    "__bibtex__",
}


def read_core_metadata() -> dict[str, object]:
    content = CORE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(CORE_PATH))
    metadata: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in METADATA_KEYS:
                metadata[target.id] = ast.literal_eval(node.value)
    missing = METADATA_KEYS.difference(metadata.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing metadata values: {missing_list}")
    return metadata


def build_readme() -> None:
    parts: list[str] = []
    for source in README_SOURCES:
        parts.append(source.read_text(encoding="utf-8").rstrip())
    output = "\n\n".join(parts).strip() + "\n"
    README_PATH.write_text(output, encoding="utf-8")


def update_pyproject(metadata: dict[str, object]) -> None:
    doc = parse(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = doc.get("project")
    if project is None:
        raise ValueError("[project] table missing from pyproject.toml")

    project["name"] = metadata["__name__"]
    project["version"] = metadata["__version__"]
    project["authors"] = metadata["__authors__"]

    urls = project.get("urls")
    if urls is None:
        urls = table()
    urls["documentation"] = metadata["__doc__"]
    urls["repository"] = metadata["__rep__"]
    project["urls"] = urls

    PYPROJECT_PATH.write_text(dumps(doc), encoding="utf-8")


def update_citation(metadata: dict[str, object]) -> None:
    bibtex = str(metadata["__bibtex__"]).strip()
    CITATION_PATH.write_text(bibtex + "\n", encoding="utf-8")


def main() -> None:
    metadata = read_core_metadata()
    build_readme()
    update_pyproject(metadata)
    update_citation(metadata)


if __name__ == "__main__":
    main()
