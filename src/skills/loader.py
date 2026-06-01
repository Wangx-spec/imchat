from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

from .base import Skill, SkillParameter
from .registry import registry

logger = logging.getLogger("skills.loader")

_DEFS_DIR = Path(__file__).parent / "defs"
_loaded = False


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal frontmatter parser for our controlled SKILL.md format.

    Supports top-level ``key: value`` scalars and a ``parameters:`` block that
    is a list of ``- name: ...`` items with indented ``key: value`` fields.
    Intentionally avoids a PyYAML dependency.
    """
    data: dict[str, Any] = {}
    lines = text.splitlines()
    i, n = 0, len(lines)

    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or raw.startswith((" ", "\t")):
            i += 1
            continue

        key, _, val = raw.partition(":")
        key = key.strip()
        val = val.strip()

        if val:
            data[key] = _coerce(val)
            i += 1
            continue

        # Block value: collect indented list items.
        items: list[dict[str, Any]] = []
        i += 1
        while i < n and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
            sub = lines[i].strip()
            if not sub:
                i += 1
                continue
            if sub.startswith("- "):
                item: dict[str, Any] = {}
                first = sub[2:].strip()
                if first and ":" in first:
                    k2, _, v2 = first.partition(":")
                    item[k2.strip()] = _coerce(v2.strip())
                items.append(item)
            elif ":" in sub and items:
                k2, _, v2 = sub.partition(":")
                items[-1][k2.strip()] = _coerce(v2.strip())
            i += 1
        data[key] = items

    return data


def _parse_frontmatter(md_path: Path) -> dict[str, Any] | None:
    try:
        content = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("[SKILL_LOAD] read failed path=%s err=%s", md_path, exc)
        return None

    if not content.startswith("---"):
        logger.warning("[SKILL_LOAD] missing frontmatter path=%s", md_path)
        return None

    end = content.find("---", 3)
    if end == -1:
        logger.warning("[SKILL_LOAD] unterminated frontmatter path=%s", md_path)
        return None

    try:
        data = _parse_simple_yaml(content[3:end])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKILL_LOAD] frontmatter parse error path=%s err=%s", md_path, exc)
        return None

    return data if isinstance(data, dict) else None


def _load_function(skill_dir: Path, script_name: str, function_name: str) -> Callable[..., Any]:
    module_path = skill_dir / "script" / f"{script_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"script not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        f"skill_{skill_dir.name.replace('-', '_')}_{script_name}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"function '{function_name}' not found in {module_path}")
    return getattr(module, function_name)


def _build_parameters(raw: Any) -> list[SkillParameter]:
    params: list[SkillParameter] = []
    for item in raw or []:
        if not isinstance(item, dict) or "name" not in item:
            continue
        params.append(
            SkillParameter(
                name=item["name"],
                type=item.get("type", "string"),
                description=item.get("description", ""),
                required=bool(item.get("required", False)),
                enum=item.get("enum"),
            )
        )
    return params


def load_all(force: bool = False) -> int:
    """Scan defs/, parse SKILL.md frontmatter, load functions and register skills.

    Returns the number of skills registered. Failures on individual skills are
    logged and skipped (graceful degradation).
    """
    global _loaded
    if _loaded and not force:
        return len(registry.select())

    if not _DEFS_DIR.exists():
        logger.warning("[SKILL_LOAD] defs dir not found: %s", _DEFS_DIR)
        _loaded = True
        return 0

    count = 0
    for skill_dir in sorted(_DEFS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue

        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            continue

        meta = _parse_frontmatter(md_path)
        if not meta or "name" not in meta:
            logger.warning("[SKILL_LOAD] skip %s: invalid frontmatter", skill_dir.name)
            continue

        script_name = meta.get("script")
        if not script_name:
            logger.warning("[SKILL_LOAD] skip %s: missing 'script'", skill_dir.name)
            continue

        function_name = meta.get("function", meta["name"])
        try:
            func = _load_function(skill_dir, script_name, function_name)
        except (FileNotFoundError, AttributeError, ImportError) as exc:
            logger.warning("[SKILL_LOAD] skip %s: %s", skill_dir.name, exc)
            continue

        registry.register(
            Skill(
                name=meta["name"],
                description=meta.get("description", ""),
                function=func,
                parameters=_build_parameters(meta.get("parameters")),
            )
        )
        count += 1
        logger.info("[SKILL_LOAD] registered skill=%s dir=%s", meta["name"], skill_dir.name)

    _loaded = True
    logger.info("[SKILL_LOAD] total=%d", count)
    return count
