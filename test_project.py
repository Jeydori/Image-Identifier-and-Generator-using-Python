import os
from PIL import Image
from unittest.mock import patch, MagicMock
from project import Fetch, checkDevice, fileCounter, imageGenerator, imageIdentifier


def test_check_device():
    device = checkDevice()
    assert device.type in ["cpu", "cuda"]


def test_file_counter():
    fetch = Fetch("prompt")
    fetch._counter = 1
    open("generated_image1.jpg", "a").close()  # create dummy file
    filename = fileCounter(fetch)
    assert filename == "generated_image2.jpg"
    os.remove("generated_image1.jpg")


@patch("project.StableDiffusionPipeline.from_pretrained")
def test_image_generator(mock_pipe):
    fetch = Fetch("prompt")
    mock_pipeline = MagicMock()
    mock_image = Image.new("RGB", (64, 64))
    mock_pipeline.return_value.__call__.return_value.images = [mock_image]
    mock_pipe.return_value = mock_pipeline

    imageGenerator(fetch)
    assert os.path.exists("generated_image1.jpg")
    os.remove("generated_image1.jpg")


@patch("project.pipeline")
@patch("project.Image.open")
@patch("builtins.input", side_effect=["0"])
def test_image_identifier(mock_input, mock_open, mock_pipeline):
    fetch = Fetch("image.jpg")
    mock_img = Image.new("RGB", (64, 64))
    mock_open.return_value.__enter__.return_value = mock_img

    mock_pipeline.return_value = MagicMock(return_value=[{"generated_text": "Caption"}])
    result = imageIdentifier(fetch)
    assert result == "Caption"
