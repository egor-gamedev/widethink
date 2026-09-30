from __future__ import annotations

from pathlib import Path

import pytest

from widethink import Context, ContextItem
from widethink.context import render_item, split_text


def test_from_text_is_a_salient_user_item() -> None:
    context = Context.from_text("We run on Raspberry Pi devices.")
    [item] = context.items
    assert item.kind == "user"
    assert item.salience == 0.9


def test_from_mapping_keeps_order() -> None:
    context = Context.from_mapping({"a": "1", "b": "2"})
    assert [item.id for item in context.items] == ["a", "b"]


def test_duplicate_ids_are_rejected() -> None:
    item = ContextItem(id="x", title="x", content="1")
    with pytest.raises(ValueError, match="duplicate"):
        Context.of(item, item)


def test_contexts_add_up() -> None:
    combined = Context.from_text("a") + Context.from_mapping({"b": "2"})
    assert combined.ids() == {"user", "b"}
    assert len(combined) == 2


def test_from_path_walks_sorted_skips_junk_and_chunks(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\nOffline first.\n", encoding="utf-8")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text("x = 1\n" * 3000, encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("junk", encoding="utf-8")
    (tmp_path / "image.py").write_bytes(b"\x00\x01binary")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "deploy.yml").write_text("region: eu-only\n", encoding="utf-8")

    context = Context.from_path(tmp_path, chunk_chars=6000)
    ids = [item.id for item in context.items]

    assert ids[:2] == ["README.md", ".github/deploy.yml"]  # top-down, sorted; dot-dirs kept
    assert not any("node_modules" in i for i in ids)
    assert "image.py" not in ids
    chunks = [i for i in ids if i.startswith("app/models.py#")]
    assert len(chunks) == 3
    assert context.get("README.md").kind == "doc"  # type: ignore[union-attr]
    assert context.get(chunks[0]).title == "app/models.py (part 1/3)"  # type: ignore[union-attr]


def test_truncation_is_announced(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    for name in "abc":
        (tmp_path / f"{name}.md").write_text(name, encoding="utf-8")
    context = Context.from_path(tmp_path, max_items=2)
    assert context.ids() == {"a.md", "b.md"}
    assert "truncated to 2 items (1 files not read)" in caplog.text


def test_from_path_accepts_a_single_file(tmp_path: Path) -> None:
    target = tmp_path / "spec.txt"
    target.write_text("Must work offline.", encoding="utf-8")
    assert Context.from_path(target).ids() == {"spec.txt"}


def test_coerce_accepts_every_documented_form(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    item = ContextItem(id="i", title="i", content="c")
    assert len(Context.coerce(None)) == 0
    assert Context.coerce("text").items[0].kind == "user"
    assert Context.coerce({"t": "c"}).ids() == {"t"}
    assert Context.coerce(item).ids() == {"i"}
    assert Context.coerce([item]).ids() == {"i"}
    assert Context.coerce(tmp_path).ids() == {"a.md"}
    assert Context.coerce(str(tmp_path)).ids() == {"a.md"}  # a string naming a path
    assert Context.coerce("no such dir, just words").items[0].kind == "user"
    same = Context.of(item)
    assert Context.coerce(same) is same


def test_split_text_respects_limit_and_lines() -> None:
    text = "".join(f"line {i}\n" for i in range(100))
    chunks = split_text(text, 50)
    assert "".join(chunks) == text
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert all(chunk.endswith("\n") for chunk in chunks)


def test_split_text_cuts_enormous_lines() -> None:
    chunks = split_text("a" * 120 + "\nb\n", 50)
    assert "".join(chunks) == "a" * 120 + "\nb\n"
    assert all(len(chunk) <= 50 for chunk in chunks)
    with pytest.raises(ValueError, match="positive"):
        split_text("x", 0)


def test_digest_lists_most_salient_items_in_original_order() -> None:
    items = [ContextItem(id=f"i{n}", title=f"t{n}", content="x", salience=n / 10) for n in range(5)]
    digest = Context.of(*items).digest(limit=2)
    assert digest.splitlines()[:2] == ["- [i3] t3 (note, 1 lines)", "- [i4] t4 (note, 1 lines)"]
    assert "3 more items" in digest
    assert Context().digest() == "(no context was provided)"


def test_render_item_truncates_visibly_and_escapes_titles() -> None:
    item = ContextItem(id="a", title='say "hi"', content="x" * 50)
    rendered = render_item(item, 10)
    assert "[... truncated 40 chars]" in rendered
    assert 'title="say &quot;hi&quot;"' in rendered
