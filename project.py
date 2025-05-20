from transformers import pipeline
from PIL import Image
import sys
from diffusers import StableDiffusionPipeline
import torch
import os


class Fetch:
    def __init__(self, prompt: str):
        self._prompt = prompt
        self._device = checkDevice()
        self._counter = 1

    @property
    def prompt(self):
        return self._prompt

    @property
    def device(self):
        return self._device

    @property
    def counter(self):
        return self._counter

def main():
    while True:
        try:
            task: int = int(input("Type '0' for Image Identification | Type '1' for Image Generation: \nInput: ").strip())

            if task == 0:
                prompt: str = input("Input filename (make sure that image is saved in the root folder): ").strip()
                try:
                    if prompt.endswith("jpg") or prompt.endswith("png"):
                        fetch = Fetch(prompt)
                        answer = imageIdentifier(fetch)
                        print(f"Result: {answer}")
                        break
                    else:
                        print("Not a valid file type")
                        pass
                except ValueError:
                    print("Not a valid file type")
                    pass
            elif task == 1:
                try:
                    prompt: str = input("Generate a/an? ").strip()
                    fetch = Fetch(prompt)
                    imageGenerator(fetch)
                    print("Done generating, locate the image in the root folder")
                    break
                except ValueError:
                    print("Not a valid input")
                    pass
            else:
                print("Not a valid input")
                pass
        except ValueError:
            print("Not a valid input")
            pass

def checkDevice():
    if torch.cuda.is_available():
        print("NVIDIA GPU detected. Using CUDA.")
        return torch.device("cuda")
    else:
        print("No GPU found. Using CPU.")
        return torch.device("cpu")


def imageIdentifier(fetch: Fetch):
    print("Setting up the environment...")
    while True:
        try:
            with Image.open(fetch._prompt) as img:
                image = img.convert("RGB")
                mode: int = int(input("Type '0' for captioning or '1' for Q&A: ").strip())

                if mode == 0:
                    itt_pipeline = pipeline(
                        "image-to-text",
                        model="Salesforce/blip-image-captioning-base",
                        device=0 if fetch._device.type == "cuda" else -1
                    )
                    print("Generating image caption...")
                    result = itt_pipeline(image)
                    return result[0]['generated_text']

                elif mode == 1:
                    vqa_pipeline = pipeline(
                        "visual-question-answering",
                        model="Salesforce/blip-vqa-base",
                        tokenizer="Salesforce/blip-vqa-base",
                        device=0 if fetch._device.type == "cuda" else -1
                    )
                    question: str = input("Ask a question about the image: ")
                    print("Generating answer...")
                    result = vqa_pipeline({"image": image, "question": question})
                    return result[0]['answer']

                else:
                    print("Invalid mode selected.")
                    pass

        except FileNotFoundError:
            print("File does not exist")
            pass


def imageGenerator(fetch: Fetch):
    print("Setting up the environment...")
    ig_pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if fetch._device.type == "cuda" else torch.float32
    )
    ig_pipeline.to(fetch._device)

    print("Image generating, this will take a while, please wait...")
    image = ig_pipeline(fetch._prompt).images[0]
    filename = fileCounter(fetch)
    image.save(filename)
    print(f"Image saved as: {filename}")


def fileCounter(fetch: Fetch):
    while True:
        filename = f"generated_image{fetch._counter}.jpg"
        if os.path.exists(filename):
            fetch._counter += 1
        else:
            return filename

if __name__ == "__main__":
    main()
