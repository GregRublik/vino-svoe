import io
from types import SimpleNamespace

from PIL import Image

from config import settings
from services.bottle_detector import BottleDetectionService


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self.boxes)]


def _image_bytes():
    image = Image.new("RGB", (100, 100), "red")
    for x in range(60, 100):
        for y in range(100):
            image.putpixel((x, y), (0, 0, 255))
    output = io.BytesIO()
    image.save(output, format="WEBP")
    return output.getvalue()


def test_prepare_chooses_box_with_highest_detection_score(monkeypatch):
    boxes = SimpleNamespace(
        conf=[0.55, 0.91],
        xyxy=[[0, 0, 30, 100], [60, 0, 100, 100]],
    )
    model = FakeModel(boxes)
    monkeypatch.setattr(settings, "yolo_crop_margin", 0.0)

    prepared = BottleDetectionService(model=model).prepare(_image_bytes())

    cropped = Image.open(io.BytesIO(prepared.data)).convert("RGB")
    assert cropped.size == (40, 100)
    assert cropped.getpixel((20, 50)) == (0, 0, 255)
    assert model.calls[0]["conf"] == settings.yolo_detection_threshold


def test_prepare_falls_back_to_original_when_no_boxes():
    model = FakeModel(SimpleNamespace(conf=[], xyxy=[]))

    prepared = BottleDetectionService(model=model).prepare(_image_bytes())

    assert prepared.data == _image_bytes()
