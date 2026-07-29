from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tree_sitter as ts
from tree_sitter_language_pack import get_language, get_parser

TagKind = Literal["def", "ref"]

# extension -> tree-sitter language name. Grow by adding an entry + a query file.
_EXT_TO_LANG = {".py": "python"}

_QUERY_DIR = Path(__file__).parent / "queries"


@dataclass(frozen=True)
class Tag:
    name: str
    kind: TagKind
    path: str
    line: int


def language_for(path: str) -> str | None:
    return _EXT_TO_LANG.get(Path(path).suffix)


def _load_query(lang: str) -> ts.Query:
    scm = (_QUERY_DIR / f"{lang}.scm").read_text(encoding="utf-8")
    return ts.Query(get_language(lang), scm)


def extract_tags(path: str, source: bytes) -> list[Tag]:
    lang = language_for(path=path)
    if lang is None:
        return []

    tree = get_parser(lang).parse(source)
    cursor = ts.QueryCursor(_load_query(lang))

    tags: list[Tag] = []

    for capture_name, nodes in cursor.captures(tree.root_node).items():
        if capture_name.startswith("definition"):
            kind: TagKind = "def"
        elif capture_name.startswith("reference"):
            kind = "ref"
        else:
            continue
        for node in nodes:
            name = source[node.start_byte : node.end_byte].decode("utf-8", "replace")
            tags.append(
                Tag(name=name, kind=kind, path=path, line=node.start_point[0] + 1)
            )
    return tags
