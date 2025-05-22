import pytest
import os
from PIL import Image
from project import Fetch, checkDevice, fileCounter, imageGenerator, imageIdentifier


def test_checkDevice():
    device = checkDevice()
    assert device.type in ["cpu", "cuda"]


def test_fileCounter():
    fetch = Fetch("prompt")
    fetch._counter = 1
    open("generated_image1.jpg", "a").close()  # create dummy file
    filename = fileCounter(fetch)
    assert filename == "generated_image2.jpg"
    os.remove("generated_image1.jpg")


def test_imageGenerator(monkeypatch):
    dummy_image = Image.new("RGB", (64, 64))

    class DummyPipeline:
        def to(self, device):
            return self
        def __call__(self, prompt):
            return type("Result", (object,), {"images": [dummy_image]})

    class DummyPipelineClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyPipeline()

    monkeypatch.setattr("project.StableDiffusionPipeline", DummyPipelineClass)

    fetch = Fetch("a cool cyberpunk dog")
    imageGenerator(fetch)
    filepath = os.path.join("generated_imgs", "generated_image1.jpg")
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_image_identifier(monkeypatch):
    dummy_image = Image.new("RGB", (64, 64))

    monkeypatch.setattr("builtins.input", lambda _: "0")

    class DummyImageOpen:
        def __enter__(self): return dummy_image
        def __exit__(self, *args): pass
    monkeypatch.setattr("project.Image.open", lambda _: DummyImageOpen())

    class DummyPipeline:
        def __call__(self, image): return [{"generated_text": "Caption"}]
    monkeypatch.setattr("project.pipeline", lambda *args, **kwargs: DummyPipeline())

    fetch = Fetch("image.jpg")
    result = imageIdentifier(fetch)
    assert result == "Caption"

