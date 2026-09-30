from __future__ import annotations

import pytest

from widethink import ThoughtTree


def build() -> ThoughtTree:
    tree = ThoughtTree("  Add   JWT auth  ")
    a = tree.add("n0", kind="solution", origin="proposal", label="A", link=1.7, value=-3)
    tree.add(a.id, kind="context", origin="proposal", label="A1", grounding=["x", "x", "y"])
    tree.add("n0", kind="risk", origin="proposal", label="B")
    return tree


def test_root_label_is_normalized_and_ids_are_sequential() -> None:
    tree = build()
    assert tree.root.label == "Add JWT auth"
    assert [node.id for node in tree.nodes()] == ["n0", "n1", "n2", "n3"]


def test_add_clamps_scores_and_dedupes_grounding() -> None:
    tree = build()
    assert tree["n1"].link == 1.0
    assert tree["n1"].value == -1.0
    assert tree["n2"].grounding == ["x", "y"]
    assert tree["n2"].depth == 2


def test_navigation() -> None:
    tree = build()
    assert [node.label for node in tree.path("n2")] == ["Add JWT auth", "A", "A1"]
    assert [node.id for node in tree.walk()] == ["n0", "n1", "n2", "n3"]
    assert [node.id for node in tree.children("n0")] == ["n1", "n3"]
    assert tree.parent("n2").id == "n1"  # type: ignore[union-attr]
    assert tree.parent("n0") is None
    assert tree.max_depth == 2
    assert "n3" in tree
    assert len(tree) == 4


def test_frontier_is_the_open_ideas() -> None:
    tree = build()
    tree["n1"].status = "expanded"
    tree["n0"].status = "expanded"
    assert [node.id for node in tree.frontier()] == ["n2", "n3"]


def test_long_labels_are_clipped() -> None:
    tree = ThoughtTree("x" * 1000)
    assert len(tree.root.label) == 300
    assert tree.root.label.endswith("…")


def test_round_trip_through_nodes() -> None:
    tree = build()
    rebuilt = ThoughtTree.from_nodes(tree.nodes())
    assert [n.model_dump() for n in rebuilt.nodes()] == [n.model_dump() for n in tree.nodes()]
    rebuilt["n1"].label = "changed"
    assert tree["n1"].label == "A"  # deep copy


def test_from_nodes_requires_the_root_first() -> None:
    tree = build()
    with pytest.raises(ValueError, match="root"):
        ThoughtTree.from_nodes(tree.nodes()[1:])


def test_node_text_and_thought_flag() -> None:
    tree = build()
    node = tree["n1"]
    assert node.text() == "A"
    node.detail = "detail"
    assert node.text() == "A. detail"
    assert not node.thought
    node.status = "failed"
    assert node.thought
