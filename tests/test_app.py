from collections import Counter

import app


def test_normalize_name_strips_honorifics_and_possessives():
    assert app.normalize_name("Mr. Darcy's") == "Darcy"
    assert app.normalize_name("Lady Catherine") == "Catherine"


def test_normalize_name_keeps_surface_form_of_nicknames():
    assert app.normalize_name("Bob Sutton") == "Bob Sutton"
    assert app.normalize_name("Liz") == "Liz"


def test_normalize_name_title_cases():
    assert app.normalize_name("elizabeth bennet") == "Elizabeth Bennet"


def test_is_likely_valid_person_name():
    assert app.is_likely_valid_person_name("Alice")
    assert not app.is_likely_valid_person_name("")
    assert not app.is_likely_valid_person_name("Monday")
    assert not app.is_likely_valid_person_name("Al")
    assert not app.is_likely_valid_person_name("R2D2")


def test_strip_possessive():
    assert app.strip_possessive("Alice's") == "Alice"
    assert app.strip_possessive("Alice’s") == "Alice"
    assert app.strip_possessive("Alice") == "Alice"


def test_chunk_text_within_target_preserves_content():
    paras = ["a" * 80, "b" * 80, "c" * 80]
    text = "\n\n".join(paras)
    chunks = app.chunk_text(text, target_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "\n\n".join(chunks) == text


def test_chunk_text_splits_oversized_paragraph():
    text = "y" * 250
    chunks = app.chunk_text(text, target_chars=100)
    assert "".join(chunks) == text
    assert max(len(c) for c in chunks) <= 100


def test_compute_paragraph_spans():
    text = "First para.\n\nSecond para.\n\nThird."
    spans = app.compute_paragraph_spans(text)
    assert [text[s:e] for s, e in spans] == ["First para.", "Second para.", "Third."]


def test_compute_scene_map_blank_line_break():
    text = "A.\n\n\n\nB."
    spans = app.compute_paragraph_spans(text)
    scene_map = app.compute_scene_map(text, spans)
    assert scene_map[0] == 0
    assert scene_map[1] == 1


def test_compute_scene_map_paragraph_count_break():
    text = "\n\n".join(f"Paragraph {i}." for i in range(5))
    spans = app.compute_paragraph_spans(text)
    scene_map = app.compute_scene_map(text, spans)
    assert [scene_map[i] for i in range(5)] == [0, 0, 0, 1, 1]


def test_compute_chunk_layout_straddling_paragraph():
    text = "aaaa\n\n" + "b" * 300
    chunks = app.chunk_text(text, target_chars=100)
    spans = app.compute_paragraph_spans(text)
    slices, indices = app.compute_chunk_layout(text, spans, chunks)
    assert indices == [[0], [1], [1], [1]]
    assert all(len(s) == len(i) for s, i in zip(slices, indices))


def test_should_merge_nickname_with_full_name():
    assert app.should_merge_name_pair("Bob", "Robert Sutton", {})


def test_should_merge_subname():
    assert app.should_merge_name_pair("Alice", "Alice Harper", {})


def test_should_not_merge_distinct_names():
    assert not app.should_merge_name_pair("Alice Harper", "Diana Cole", {})
    assert not app.should_merge_name_pair("Alice", "Diana", {})


def test_choose_canonical_prefers_fuller_frequent_name():
    counts = {"Alice": 30, "Alice Harper": 5}
    assert app.choose_canonical_name("Alice", "Alice Harper", counts) == "Alice Harper"


def test_choose_canonical_rejects_rare_fuller_name():
    counts = {"Alice": 30, "Alice Harper": 1}
    assert app.choose_canonical_name("Alice", "Alice Harper", counts) == "Alice"


def test_build_alias_map_merges_cluster():
    counts = {"Bob": 10, "Robert Sutton": 4, "Diana": 6}
    alias_map = app.build_alias_map(counts, {})
    assert alias_map["Bob"] == "Robert Sutton"
    assert alias_map["Robert Sutton"] == "Robert Sutton"
    assert alias_map["Diana"] == "Diana"


def test_edge_confidence_reflects_signal_precision():
    weak = app.edge_confidence(Counter({"same_scene": 1}))
    medium = app.edge_confidence(Counter({"co_mention": 2, "same_paragraph": 1}))
    strong = app.edge_confidence(Counter({"dialogue": 2, "dependency": 1}))
    assert 0 < weak < medium < strong < 1.0
    assert app.edge_confidence(Counter({"co_mention": 3})) == 0.45
    assert app.edge_confidence(Counter({"dependency": 0})) == 0.0


def test_jaccard_similarity():
    assert app.jaccard_similarity({"a", "b"}, {"b", "c"}) == 1 / 3
    assert app.jaccard_similarity(set(), {"a"}) == 0.0
