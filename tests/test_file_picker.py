from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = PROJECT_ROOT / "src/templates/index.html"
CAMERA_JS = PROJECT_ROOT / "src/static/js/camera.js"


def test_file_picker_is_browser_managed_and_not_directory_bound():
    html = INDEX_HTML.read_text(encoding="utf-8")
    javascript = CAMERA_JS.read_text(encoding="utf-8")

    assert 'type="file"' in html
    assert 'id="galleryBtn"' in html
    assert "webkitdirectory" not in html
    assert "directory" not in html
    assert "startIn: 'downloads'" in javascript
    assert "Датасет" not in html
    assert "Датасет" not in javascript
    assert "/home/" not in html
    assert "/home/" not in javascript
    assert 'value="' not in html
    assert "this.files[0]" in javascript


def test_selected_file_is_uploaded_as_bytes_without_a_local_path():
    javascript = CAMERA_JS.read_text(encoding="utf-8")

    assert "formData.append('photo', selectedFile)" in javascript
    assert "selectedFile.path" not in javascript
    assert "selectedFile.fullPath" not in javascript


def test_picker_uses_only_browser_file_objects():
    javascript = CAMERA_JS.read_text(encoding="utf-8")

    assert "fileHandle.getFile()" in javascript
    assert "fileInput.click()" in javascript
    assert "selectedFile.name" in javascript
    assert "selectedFile.path" not in javascript
