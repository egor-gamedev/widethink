"""The user's context: what the harness can look at besides the task itself.

Hidden requirements live here - in the code, the data model, the README, a
sentence the user wrote. The context is a flat list of items; large files are
split into chunks so that each step can be shown just the relevant parts.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("widethink")

#: Default salience by kind: what the user says outweighs what the code implies.
KIND_SALIENCE: dict[str, float] = {"user": 0.9, "doc": 0.7, "note": 0.6, "file": 0.5}

DEFAULT_INCLUDE: tuple[str, ...] = (
    "*.py",
    "*.pyi",
    "*.md",
    "*.rst",
    "*.txt",
    "*.toml",
    "*.cfg",
    "*.ini",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.js",
    "*.jsx",
    "*.ts",
    "*.tsx",
    "*.go",
    "*.rs",
    "*.java",
    "*.kt",
    "*.cs",
    "*.rb",
    "*.php",
    "*.swift",
    "*.c",
    "*.h",
    "*.cpp",
    "*.hpp",
    "*.sql",
    "*.sh",
    "*.proto",
    "*.graphql",
    "*.html",
    "*.css",
    "Dockerfile",
    "Makefile",
    "*.env.example",
)

DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        "target",
        "coverage",
        ".next",
        ".cache",
        ".idea",
        ".vscode",
    }
)

_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt"})


class ContextItem(BaseModel):
    """One piece of the user's context."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Stable short identifier, cited by the model as evidence (e.g. ``app/models.py#2``)."""
    title: str
    content: str
    kind: str = "note"
    """Free-form; ``user``, ``doc``, ``file`` and ``note`` get default saliences."""
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    """How much the item matters a priori (brain: significance of a memory trace)."""


class Context(BaseModel):
    """An ordered collection of context items with unique ids."""

    model_config = ConfigDict(frozen=True)

    items: tuple[ContextItem, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: str) -> ContextItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def ids(self) -> set[str]:
        return {item.id for item in self.items}

    def __add__(self, other: Context) -> Context:
        return Context.of(*self.items, *other.items)

    # ------------------------------------------------------------------ constructors

    @classmethod
    def of(cls, *items: ContextItem) -> Context:
        """Build a context, rejecting duplicate ids."""
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise ValueError(f"duplicate context item id: {item.id!r}")
            seen.add(item.id)
        return cls(items=tuple(items))

    @classmethod
    def from_text(cls, text: str, *, title: str = "User context", kind: str = "user") -> Context:
        """A context made of one free-text item (for example, what the user said)."""
        return cls.of(
            ContextItem(id=kind, title=title, content=text, kind=kind, salience=_salience(kind))
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], *, kind: str = "note") -> Context:
        """One item per ``title -> content`` pair."""
        return cls.of(
            *(
                ContextItem(
                    id=title, title=title, content=content, kind=kind, salience=_salience(kind)
                )
                for title, content in mapping.items()
            )
        )

    @classmethod
    def from_path(
        cls,
        root: str | os.PathLike[str],
        *,
        include: Sequence[str] = DEFAULT_INCLUDE,
        exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
        max_file_bytes: int = 256_000,
        chunk_chars: int = 6000,
        max_items: int = 400,
    ) -> Context:
        """Load a project directory (or a single file) as context.

        Files are read in a stable (sorted) order, binary and oversized files are
        skipped, and long files are split at line boundaries into chunks of about
        ``chunk_chars`` characters.
        """
        base = Path(root)
        excluded = frozenset(exclude_dirs)
        files = [base] if base.is_file() else _walk(base, include, excluded)
        items: list[ContextItem] = []
        unread = 0
        for position, path in enumerate(files):
            if len(items) >= max_items:
                unread = len(files) - position
                break
            text = _read_text(path, max_file_bytes)
            if text is None or not text.strip():
                continue
            rel = path.name if base.is_file() else path.relative_to(base).as_posix()
            kind = "doc" if path.suffix.lower() in _DOC_SUFFIXES else "file"
            chunks = split_text(text, chunk_chars)
            for index, chunk in enumerate(chunks, start=1):
                many = len(chunks) > 1
                items.append(
                    ContextItem(
                        id=f"{rel}#{index}" if many else rel,
                        title=f"{rel} (part {index}/{len(chunks)})" if many else rel,
                        content=chunk,
                        kind=kind,
                        salience=_salience(kind),
                    )
                )
        if unread or len(items) > max_items:
            logger.warning(
                "context of %s truncated to %d items (%d files not read); raise max_items "
                "or narrow the include patterns",
                base,
                max_items,
                unread,
            )
        return cls.of(*items[:max_items])

    @classmethod
    def coerce(cls, value: ContextLike) -> Context:
        """Accept the many convenient forms of context the public API allows.

        A path (``Path`` or ``os.PathLike``) is loaded with :meth:`from_path`. A
        string that names an existing file or directory is loaded the same way;
        any other string is free text from the user.
        """
        if value is None:
            return cls()
        if isinstance(value, Context):
            return value
        if isinstance(value, ContextItem):
            return cls.of(value)
        if isinstance(value, os.PathLike):
            return cls.from_path(value)
        if isinstance(value, str):
            return cls.from_path(value) if _names_a_path(value) else cls.from_text(value)
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        return cls.of(*value)

    # ------------------------------------------------------------------ rendering

    def digest(self, limit: int = 80) -> str:
        """One line per item - enough for the model to know what exists."""
        if not self.items:
            return "(no context was provided)"
        ranked = sorted(self.items, key=lambda item: -item.salience)[:limit]
        order = {item.id: position for position, item in enumerate(self.items)}
        lines = [
            f"- [{item.id}] {item.title} ({item.kind}, {item.content.count(chr(10)) + 1} lines)"
            for item in sorted(ranked, key=lambda item: order[item.id])
        ]
        if len(self.items) > limit:
            lines.append(f"- ... and {len(self.items) - limit} more items")
        return "\n".join(lines)


ContextLike: TypeAlias = (
    Context
    | ContextItem
    | str
    | os.PathLike[str]
    | Sequence[ContextItem]
    | Mapping[str, str]
    | None
)


def render_item(item: ContextItem, max_chars: int) -> str:
    """Render an item for a prompt, truncating long content visibly."""
    content = item.content
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n[... truncated {len(item.content) - max_chars} chars]"
    title = item.title.replace('"', "&quot;")
    return f'<item id="{item.id}" kind="{item.kind}" title="{title}">\n{content}\n</item>'


def split_text(text: str, chunk_chars: int) -> list[str]:
    """Split text into chunks of at most ``chunk_chars``, preferring line boundaries."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line
        while len(line) > chunk_chars:  # a single enormous line
            if current:
                chunks.append("".join(current))
                current, size = [], 0
            chunks.append(line[:chunk_chars])
            line = line[chunk_chars:]
        if size + len(line) > chunk_chars and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _names_a_path(text: str) -> bool:
    if not text or "\n" in text or len(text) > 4096:
        return False
    try:
        return Path(text).exists()
    except (OSError, ValueError):
        return False


def _salience(kind: str) -> float:
    return KIND_SALIENCE.get(kind, 0.5)


def _walk(base: Path, include: Sequence[str], excluded: frozenset[str]) -> list[Path]:
    found: list[Path] = []
    for directory, subdirs, files in os.walk(base):
        subdirs[:] = sorted(d for d in subdirs if d not in excluded)
        for name in sorted(files):
            if any(fnmatch.fnmatch(name, pattern) for pattern in include):
                found.append(Path(directory) / name)
    return found


def _read_text(path: Path, max_bytes: int) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    # Normalize line endings: prompts (and the hashes of recorded requests) must not
    # depend on the operating system the project was checked out on.
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
