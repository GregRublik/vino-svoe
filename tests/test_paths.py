from config import PROJECT_ROOT, resolve_project_path


def test_relative_project_path_does_not_depend_on_current_directory(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    assert resolve_project_path("data/images") == PROJECT_ROOT / "data/images"


def test_absolute_project_path_is_preserved(tmp_path):
    assert resolve_project_path(tmp_path / "catalog.json") == tmp_path / "catalog.json"
