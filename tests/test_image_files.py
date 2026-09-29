from pathlib import Path

from utils.image_files import deduplicate_files


def test_deduplicate_files_keeps_first_and_preserves_different_content(tmp_path: Path):
    first = tmp_path / "first.webp"
    duplicate = tmp_path / "duplicate.webp"
    different = tmp_path / "different.webp"
    first.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    different.write_bytes(b"other")

    unique, duplicate_count = deduplicate_files([first, duplicate, different])

    assert unique == [first, different]
    assert duplicate_count == 1
