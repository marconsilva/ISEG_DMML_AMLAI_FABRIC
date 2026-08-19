"""Audit Fabric notebook runtime metadata, imports, and dataset references."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATTERN = "**/notebook-content.py"
EXPECTED_NOTEBOOKS = 142
FABRIC_BUILT_INS = {
    "python",
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "pyspark",
    "mlflow",
}
IMPORT_TO_PACKAGE = {
    "IPython": "ipython",
    "azure": "azure-search-documents",
    "category_encoders": "category-encoders",
    "charset_normalizer": "charset-normalizer",
    "fuzzywuzzy": "fuzzywuzzy",
    "graphviz": "graphviz",
    "imblearn": "imbalanced-learn",
    "ipywidgets": "ipywidgets",
    "keras": "tensorflow",
    "learntools": "learntools",
    "lightgbm": "lightgbm",
    "matplotlib": "matplotlib",
    "mlflow": "mlflow",
    "mlxtend": "mlxtend",
    "mpl_toolkits": "matplotlib",
    "numpy": "numpy",
    "openai": "openai",
    "pandas": "pandas",
    "pmdarima": "pmdarima",
    "pylab": "matplotlib",
    "pyspark": "pyspark",
    "requests": "requests",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "spacy": "spacy",
    "statsmodels": "statsmodels",
    "sympy": "sympy",
    "synapse": "synapseml",
    "tensorflow": "tensorflow",
    "tensorflow_hub": "tensorflow-hub",
    "xgboost": "xgboost",
}
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import|"
    r"import\s+(.+))"
)
DATA_RE = re.compile(
    r"""["'](
        (?:/lakehouse/default/)?(?:Files|Tables)/[^"']+
        |/lakehouse/default/Files/[^"']+
        |[^"']+\.(?:csv|json|parquet|xlsx?|txt|zip|pkl|pickle|tsv|data|gz)
    )["']""",
    re.IGNORECASE | re.VERBOSE,
)
NOTEBOOK_MARKER_RE = re.compile(r"^# (CELL|MARKDOWN|METADATA) \*+\s*$")
NON_PYTHON_CELL_RE = re.compile(r"^%%(?:sql|pyspark|spark|scala|r)\b", re.I)
LINE_MAGIC_RE = re.compile(r"^\s*[%!][A-Za-z_]")


def read_notebooks() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8-sig", errors="replace"))
        for path in sorted(ROOT.glob(NOTEBOOK_PATTERN))
    ]


def collect_imports(notebooks: list[tuple[Path, str]]) -> Counter[str]:
    imports: Counter[str] = Counter()
    for _, text in notebooks:
        seen: set[str] = set()
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            match = IMPORT_RE.match(line.split("#", 1)[0])
            if not match:
                continue
            names = [match.group(1)] if match.group(1) else match.group(2).split(",")
            for name in names:
                module = name.strip().split()[0].split(".")[0]
                if module.isidentifier():
                    seen.add(module)
        imports.update(seen)
    return imports


def collect_data_references(notebooks: list[tuple[Path, str]]) -> Counter[str]:
    references: Counter[str] = Counter()
    for _, text in notebooks:
        references.update(match.group(1) for match in DATA_RE.finditer(text))
    return references


def validate_notebook_metadata(
    notebooks: list[tuple[Path, str]],
) -> list[str]:
    """Return invalid embedded Fabric metadata blocks."""
    invalid = []
    for path, text in notebooks:
        lines = text.splitlines()
        line_index = 0
        while line_index < len(lines):
            if not lines[line_index].startswith("# META "):
                line_index += 1
                continue

            start_line = line_index + 1
            block = []
            while (
                line_index < len(lines)
                and lines[line_index].startswith("# META ")
            ):
                block.append(lines[line_index][7:])
                line_index += 1
            try:
                json.loads("\n".join(block))
            except json.JSONDecodeError as error:
                invalid.append(
                    f"{path.relative_to(ROOT)}:{start_line}: {error.msg}"
                )
    return invalid


def validate_notebook_syntax(
    notebooks: list[tuple[Path, str]],
) -> list[str]:
    """Parse Python cells while respecting Fabric cell and magic syntax."""
    invalid = []
    for path, text in notebooks:
        lines = text.splitlines()
        line_index = 0
        while line_index < len(lines):
            marker = NOTEBOOK_MARKER_RE.match(lines[line_index])
            if not marker or marker.group(1) != "CELL":
                line_index += 1
                continue

            start_line = line_index + 2
            line_index += 1
            cell = []
            while (
                line_index < len(lines)
                and not NOTEBOOK_MARKER_RE.match(lines[line_index])
            ):
                cell.append(lines[line_index])
                line_index += 1

            first_line = next(
                (line.strip() for line in cell if line.strip()), ""
            )
            if NON_PYTHON_CELL_RE.match(first_line):
                continue
            source = "\n".join(
                "" if LINE_MAGIC_RE.match(line) else line for line in cell
            )
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as error:
                error_line = start_line + (error.lineno or 1) - 1
                invalid.append(
                    f"{path.relative_to(ROOT)}:{error_line}: {error.msg}"
                )
    return invalid


def declared_local_packages() -> set[str]:
    path = ROOT / "requirements-local.txt"
    packages = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            if value.lower().endswith(".whl"):
                packages.add(Path(value).name.split("-", 1)[0].lower())
            else:
                packages.add(re.split(r"[<>=!~\[]", value, maxsplit=1)[0].lower())
    return packages


def fabric_environment_packages() -> list[str]:
    path = (
        ROOT
        / "Envs"
        / "genaienv.Environment"
        / "Libraries"
        / "PublicLibraries"
        / "environment.yml"
    )
    packages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value.startswith("- ") and value not in {"- pip", "- pip:"}:
            packages.append(re.split(r"[<>=!~\[]", value[2:], maxsplit=1)[0].lower())
    return packages


def fabric_environment_logical_ids() -> dict[str, str]:
    """Return logical environment IDs mapped to their display names."""
    ids: dict[str, str] = {}
    for platform_file in sorted((ROOT / "Envs").glob("*.Environment/.platform")):
        try:
            data = json.loads(platform_file.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        logical_id = data.get("config", {}).get("logicalId")
        display_name = data.get("metadata", {}).get(
            "displayName", platform_file.parent.name
        )
        if logical_id:
            ids[logical_id] = display_name
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="also report optional local modules that are not importable",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    notebooks = read_notebooks()
    imports = collect_imports(notebooks)
    data_refs = collect_data_references(notebooks)
    environment_ids = Counter(
        match.group(1)
        for _, text in notebooks
        for match in re.finditer(r'"environmentId"\s*:\s*"([^"]+)"', text)
    )
    env_logical_ids = fabric_environment_logical_ids()
    errors = []
    warnings = []

    if len(notebooks) != EXPECTED_NOTEBOOKS:
        errors.append(
            f"expected {EXPECTED_NOTEBOOKS} notebooks, found {len(notebooks)}"
        )

    missing_platform = [
        str(path.relative_to(ROOT))
        for path, _ in notebooks
        if not (path.parent / ".platform").is_file()
    ]
    if missing_platform:
        errors.append(f"{len(missing_platform)} notebooks have no .platform metadata")

    platform_files = sorted(ROOT.glob("**/.platform"))
    invalid_platform = []
    missing_logical_ids = []
    invalid_logical_ids = []
    logical_ids: dict[str, list[str]] = {}
    for path in platform_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid_platform.append(str(path.relative_to(ROOT)))
            continue

        relative_path = str(path.relative_to(ROOT))
        logical_id = data.get("config", {}).get("logicalId")
        if not logical_id:
            missing_logical_ids.append(relative_path)
            continue
        try:
            uuid.UUID(logical_id)
        except (ValueError, AttributeError):
            invalid_logical_ids.append(f"{relative_path}: {logical_id!r}")
            continue
        logical_ids.setdefault(logical_id, []).append(relative_path)
    if invalid_platform:
        errors.append(f"{len(invalid_platform)} .platform files are invalid JSON")
    if missing_logical_ids:
        errors.append(
            f"{len(missing_logical_ids)} .platform files have no logicalId"
        )
    if invalid_logical_ids:
        errors.append(
            f"{len(invalid_logical_ids)} .platform files have invalid logicalIds"
        )
    duplicate_logical_ids = {
        logical_id: paths
        for logical_id, paths in logical_ids.items()
        if len(paths) > 1
    }
    if duplicate_logical_ids:
        errors.append(
            f"{len(duplicate_logical_ids)} logicalId values are duplicated"
        )

    invalid_notebook_metadata = validate_notebook_metadata(notebooks)
    if invalid_notebook_metadata:
        errors.append(
            f"{len(invalid_notebook_metadata)} notebook metadata blocks "
            "are invalid JSON"
        )

    invalid_notebook_syntax = validate_notebook_syntax(notebooks)
    if invalid_notebook_syntax:
        errors.append(
            f"{len(invalid_notebook_syntax)} notebook Python cells "
            "have invalid syntax"
        )

    environment_dirs = sorted((ROOT / "Envs").glob("*.Environment"))
    missing_environment_metadata = [
        path.name
        for path in environment_dirs
        if not (path / ".platform").is_file()
        or not (path / "Setting" / "Sparkcompute.yml").is_file()
    ]
    if missing_environment_metadata:
        errors.append(
            "incomplete Fabric environments: "
            + ", ".join(missing_environment_metadata)
        )
    runtime_versions = Counter()
    for path in environment_dirs:
        settings = path / "Setting" / "Sparkcompute.yml"
        if settings.is_file():
            match = re.search(
                r"^runtime_version:\s*(\S+)", settings.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if match:
                runtime_versions[match.group(1)] += 1
    if len(runtime_versions) > 1:
        warnings.append(
            "Fabric environments use multiple runtimes: "
            + ", ".join(sorted(runtime_versions))
        )

    unresolved_env_ids = sorted(
        environment_id
        for environment_id in environment_ids
        if environment_id not in env_logical_ids
    )
    if unresolved_env_ids:
        warnings.append(
            f"{sum(environment_ids[value] for value in unresolved_env_ids)} "
            f"notebook(s) reference {len(unresolved_env_ids)} environmentId "
            "value(s) that do not match an Envs logicalId. These may be "
            "workspace-scoped export IDs and require a Fabric Git sync or "
            "manual rebinding after import: "
            + ", ".join(unresolved_env_ids)
        )

    local_packages = declared_local_packages()
    unmapped_imports = sorted(
        module
        for module in imports
        if module not in sys.stdlib_module_names and module not in IMPORT_TO_PACKAGE
    )
    if unmapped_imports:
        errors.append(
            "notebook imports lack package mappings: " + ", ".join(unmapped_imports)
        )
    undeclared = sorted(
        package
        for module, package in IMPORT_TO_PACKAGE.items()
        if module in imports and package.lower() not in local_packages
    )
    if undeclared:
        errors.append("local requirements omit: " + ", ".join(undeclared))

    fabric_packages = fabric_environment_packages()
    duplicated_built_ins = sorted(FABRIC_BUILT_INS.intersection(fabric_packages))
    if duplicated_built_ins:
        errors.append(
            "Fabric environment overrides built-ins: "
            + ", ".join(duplicated_built_ins)
        )

    lakehouse_refs = sorted(
        ref for ref in data_refs if "Files/" in ref or "Tables/" in ref
    )
    if lakehouse_refs:
        warnings.append(
            f"{len(lakehouse_refs)} unique lakehouse paths require uploaded data/tables"
        )

    missing_imports = []
    if args.check_imports:
        for module in sorted(IMPORT_TO_PACKAGE.keys() & imports.keys()):
            if importlib.util.find_spec(module) is None:
                missing_imports.append(module)
        if missing_imports:
            warnings.append(
                "optional local modules not installed: " + ", ".join(missing_imports)
            )

    result = {
        "notebooks": len(notebooks),
        "platform_files": len(platform_files),
        "logical_ids": len(logical_ids),
        "duplicate_logical_ids": duplicate_logical_ids,
        "invalid_notebook_metadata": invalid_notebook_metadata,
        "invalid_notebook_syntax": invalid_notebook_syntax,
        "fabric_environments": len(environment_dirs),
        "runtime_versions": dict(sorted(runtime_versions.items())),
        "environment_logical_ids": dict(sorted(env_logical_ids.items())),
        "notebook_environment_ids": dict(sorted(environment_ids.items())),
        "imported_modules": dict(sorted(imports.items())),
        "unmapped_imports": unmapped_imports,
        "unique_data_references": len(data_refs),
        "unique_lakehouse_references": len(lakehouse_refs),
        "errors": errors,
        "warnings": warnings,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Audited {result['notebooks']} notebooks, "
            f"{result['platform_files']} platform files, "
            f"{len(imports)} imported modules, and "
            f"{result['unique_data_references']} data references."
        )
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
