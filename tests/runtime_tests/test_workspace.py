"""Tests for runtime.workspace — Workspace + SubDir routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.storage.workspace import (
    SubDirSpec,
    Workspace,
    canonical_key,
    extract_id,
    register_subdir,
    registered_specs,
    strip_id,
    strip_lang_tail,
)


# ---------------------------------------------------------------------------
# Helpers: key parsing
# ---------------------------------------------------------------------------


class TestExtractId:
    def test_simple_id(self) -> None:
        assert extract_id("Intro [LUU0EuDKgKo]") == "LUU0EuDKgKo"

    def test_no_id(self) -> None:
        assert extract_id("Intro") is None

    def test_id_with_underscore_and_dash(self) -> None:
        assert extract_id("lec [abc_DEF-123]") == "abc_DEF-123"

    def test_short_brackets_not_id(self) -> None:
        assert extract_id("vid [abc]") is None  # <6 chars

    def test_id_trailing_whitespace(self) -> None:
        assert extract_id("lec [LUU0EuDKgKo]   ") == "LUU0EuDKgKo"


class TestStripId:
    def test_strip_removes_id(self) -> None:
        assert strip_id("Intro [LUU0EuDKgKo]") == "Intro"

    def test_strip_noop_without_id(self) -> None:
        assert strip_id("Intro") == "Intro"

    def test_strip_preserves_multi_word(self) -> None:
        assert strip_id("Lec 3 Topics [LUU0EuDKgKo]") == "Lec 3 Topics"


class TestStripLangTail:
    def test_strip_en(self) -> None:
        assert strip_lang_tail("Intro.en") == "Intro"

    def test_strip_en_us(self) -> None:
        assert strip_lang_tail("Intro.en-US") == "Intro"

    def test_preserves_id(self) -> None:
        assert strip_lang_tail("Intro.en [LUU0EuDKgKo]") == "Intro [LUU0EuDKgKo]"

    def test_noop_without_lang(self) -> None:
        assert strip_lang_tail("Intro") == "Intro"


class TestCanonicalKey:
    def test_plain_stem(self) -> None:
        assert canonical_key("lec03") == "lec03"

    def test_with_extension(self) -> None:
        assert canonical_key("lec03.mp4") == "lec03"

    def test_from_path(self) -> None:
        assert canonical_key(Path("a/b/lec03.mp4")) == "lec03"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtin_specs_registered(self) -> None:
        calls = {s.call for s in registered_specs()}
        for required in ("home", "audio", "subtitle", "translation"):
            assert required in calls

    def test_register_is_idempotent_on_call(self) -> None:
        from adapters.storage.workspace import _REGISTRY  # type: ignore[attr-defined]

        original = next(s for s in _REGISTRY if s.call == "home")
        original_idx = _REGISTRY.index(original)
        before = len(registered_specs())
        try:
            register_subdir(call="home", name="", types=(".mp4",))
            assert len(registered_specs()) == before  # no new entry
        finally:
            _REGISTRY[original_idx] = original


# ---------------------------------------------------------------------------
# Workspace construction
# ---------------------------------------------------------------------------


class TestWorkspaceBasic:
    def test_constructs_all_subdirs(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path, "c")
        assert ws.home.path == tmp_path / "c"
        assert ws.audio.path == tmp_path / "c" / "zzz_audio"
        assert ws.translation.path == tmp_path / "c" / "zzz_translation"

    def test_metadata_path(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path, "c")
        assert ws.metadata_path == tmp_path / "c" / "metadata.json"

    def test_nested_course(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path, "2025-09/MIT")
        assert ws.course_path == tmp_path / "2025-09" / "MIT"

    def test_rejects_empty_course(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            Workspace(tmp_path, "")

    def test_rejects_traversal_course(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            Workspace(tmp_path, "../escape")


# ---------------------------------------------------------------------------
# SubDir.files() — read
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_ws(tmp_path: Path) -> Workspace:
    """Course with a few indexed videos and subtitles."""
    root = tmp_path / "c"
    root.mkdir()
    # home: videos at root
    (root / "lec01.mp4").touch()
    (root / "lec02.mp4").touch()
    (root / "lec03 [LUU0EuDKgKo].mp4").touch()
    # audio
    audio = root / "zzz_audio"
    audio.mkdir()
    (audio / "lec01.wav").touch()
    # subtitle — only lec01 translated
    sub = root / "zzz_subtitle"
    sub.mkdir()
    (sub / "lec01.srt").touch()
    return Workspace(tmp_path, "c")


class TestFilesAll:
    def test_all_home_files(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files()
        stems = sorted(p.stem for p in hits)
        assert "lec01" in stems
        assert "lec02" in stems
        assert "lec03 [LUU0EuDKgKo]" in stems

    def test_all_excludes_metadata(self, seeded_ws: Workspace, tmp_path: Path) -> None:
        (tmp_path / "c" / "metadata.json").write_text("{}", encoding="utf-8")
        seeded_ws.reindex()
        stems = [p.stem for p in seeded_ws.home.files()]
        assert "metadata" not in stems

    def test_all_filters_by_types(self, seeded_ws: Workspace, tmp_path: Path) -> None:
        (tmp_path / "c" / "zzz_audio" / "notes.txt").touch()
        seeded_ws.reindex()
        stems = [p.stem for p in seeded_ws.audio.files()]
        assert "notes" not in stems


class TestFilesSingleKey:
    def test_hit_returns_single_path(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files("lec01")
        assert len(hits) == 1
        assert hits[0].stem == "lec01"

    def test_miss_skip_returns_empty(self, seeded_ws: Workspace) -> None:
        assert seeded_ws.home.files("missing") == []

    def test_miss_raise(self, seeded_ws: Workspace) -> None:
        with pytest.raises(FileNotFoundError):
            seeded_ws.home.files("missing", missing="raise")

    def test_miss_none_placeholder(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files("missing", missing="none")
        assert hits == [None]

    def test_key_with_extension(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files("lec01.mp4")
        assert len(hits) == 1

    def test_key_as_path(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files(Path("some/dir/lec01.mp4"))
        assert len(hits) == 1


class TestFilesBatch:
    def test_batch_all_hit(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files(["lec01", "lec02"])
        stems = [p.stem for p in hits]
        assert stems == ["lec01", "lec02"]

    def test_batch_skip_drops_misses(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files(["lec01", "nope", "lec02"])
        stems = [p.stem for p in hits]
        assert stems == ["lec01", "lec02"]

    def test_batch_none_preserves_alignment(self, seeded_ws: Workspace) -> None:
        hits = seeded_ws.home.files(["lec01", "nope", "lec02"], missing="none")
        assert len(hits) == 3
        assert hits[0].stem == "lec01"
        assert hits[1] is None
        assert hits[2].stem == "lec02"

    def test_batch_raise_on_first_miss(self, seeded_ws: Workspace) -> None:
        with pytest.raises(FileNotFoundError):
            seeded_ws.home.files(["lec01", "nope"], missing="raise")

    def test_invalid_missing_arg(self, seeded_ws: Workspace) -> None:
        with pytest.raises(ValueError):
            seeded_ws.home.files(["lec01"], missing="bogus")


class TestRouting:
    def test_match_by_id(self, seeded_ws: Workspace) -> None:
        # Query by stem that has different human name but same [id]
        hits = seeded_ws.home.files("DifferentName [LUU0EuDKgKo]")
        assert len(hits) == 1
        assert "LUU0EuDKgKo" in hits[0].stem

    def test_match_by_stripped_stem(self, seeded_ws: Workspace) -> None:
        # File stored as "lec03 [LUU0EuDKgKo].mp4"; query as "lec03" matches
        # after stripping id on both sides.
        hits = seeded_ws.home.files("lec03")
        assert len(hits) == 1

    def test_stat_fallback_picks_up_new_file(self, seeded_ws: Workspace, tmp_path: Path) -> None:
        # File created AFTER index build. stat fallback catches it.
        new = tmp_path / "c" / "lec99.mp4"
        new.touch()
        hits = seeded_ws.home.files("lec99")
        assert len(hits) == 1
        # subsequent lookup uses cached index
        hits2 = seeded_ws.home.files("lec99")
        assert hits2 == hits


# ---------------------------------------------------------------------------
# path_for — write
# ---------------------------------------------------------------------------


class TestPathFor:
    def test_default_suffix(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path, "c")
        p = ws.translation.path_for("lec01")
        assert p.name == "lec01.json"
        assert p.parent == tmp_path / "c" / "zzz_translation"

    def test_override_suffix(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path, "c")
        p = ws.translation.path_for("lec01", suffix=".bak")
        assert p.name == "lec01.bak"

    def test_accepts_path_input(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path, "c")
        p = ws.translation.path_for(Path("x/y/lec01.mp4"))
        assert p.name == "lec01.json"

    def test_truncate_stem_with_id(self, tmp_path: Path) -> None:
        # Register a test subdir with truncate
        register_subdir(
            call="_test_trunc",
            name="zzz_test_trunc",
            types=(".mp4",),
            default_suffix=".mp4",
            truncate_stem=30,
        )
        ws = Workspace(tmp_path, "c")
        long_stem = "A" * 100 + " [LUU0EuDKgKo]"
        p = ws.get_subdir("_test_trunc").path_for(long_stem, suffix=".mp4")
        # id preserved; body truncated
        assert p.stem.endswith("[LUU0EuDKgKo]")
        assert len(p.stem) <= 30

    def test_strip_id_on_write(self, tmp_path: Path) -> None:
        register_subdir(
            call="_test_strip",
            name="zzz_test_strip",
            types=(".mp4",),
            default_suffix=".mp4",
            strip_id_on_write=True,
        )
        ws = Workspace(tmp_path, "c")
        p = ws.get_subdir("_test_strip").path_for("Intro [LUU0EuDKgKo]")
        assert p.stem == "Intro"


# ---------------------------------------------------------------------------
# videos() + routes()
# ---------------------------------------------------------------------------


class TestVideosAndRoutes:
    def test_videos_lists_home_stems(self, seeded_ws: Workspace) -> None:
        vids = seeded_ws.videos()
        assert "lec01" in vids
        assert "lec02" in vids

    def test_videos_include_iterable(self, seeded_ws: Workspace) -> None:
        vids = seeded_ws.videos(include={"lec01"})
        assert vids == ["lec01"]

    def test_videos_include_predicate(self, seeded_ws: Workspace) -> None:
        vids = seeded_ws.videos(include=lambda s: s.startswith("lec0"))
        assert "lec01" in vids and "lec02" in vids

    def test_videos_exclude(self, seeded_ws: Workspace) -> None:
        vids = seeded_ws.videos(exclude={"lec01"})
        assert "lec01" not in vids
        assert "lec02" in vids

    def test_routes_single_video(self, seeded_ws: Workspace) -> None:
        r = seeded_ws.routes("lec01")
        assert r["home"] is not None
        assert r["audio"] is not None
        assert r["subtitle"] is not None
        assert r["translation"] is None  # not created yet

    def test_routes_covers_all_subdirs(self, seeded_ws: Workspace) -> None:
        r = seeded_ws.routes("lec01")
        for call in ("home", "audio", "subtitle", "translation", "material"):
            assert call in r


class TestCourses:
    def test_lists_course_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "c1").mkdir()
        (tmp_path / "c2").mkdir()
        (tmp_path / "not_a_dir.txt").touch()
        assert set(Workspace.courses(tmp_path)) == {"c1", "c2"}

    def test_empty_root(self, tmp_path: Path) -> None:
        assert Workspace.courses(tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------


class TestSubDirSpec:
    def test_frozen(self) -> None:
        spec = SubDirSpec(call="x", name="zzz_x")
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.call = "y"  # type: ignore
