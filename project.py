from transformers import pipeline
from PIL import Image
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
    print("\n\033[1mImidge\033[0m\n")
    print(f"an Image Identifier and Generator cli-based application")
    print(f"------------------------------------------------")
    print(f"Welcome! This tool lets you:")
    print(f"1. Identify and caption images.")
    print(f"2. Generate realistic images from text prompts.\n")

    while True:
        try:
            print(f"Please choose an option to get started:")
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
                        print(f"Not a valid file type")
                        pass
                except ValueError:
                    print(f"Not a valid file type")
                    pass
            elif task == 1:
                try:
                    prompt: str = input("Generate a/an? ").strip()
                    fetch = Fetch(prompt)
                    imageGenerator(fetch)
                    print(f"Done generating, locate the image in the root folder")
                    break
                except ValueError:
                    print(f"Not a valid input")
                    pass
            else:
                print(f"Not a valid input")
                pass
        except ValueError:
            print(f"Not a valid input")
            pass

def checkDevice():
    if torch.cuda.is_available():
        print(f"NVIDIA GPU detected. Using CUDA.")
        return torch.device("cuda")
    else:
        print(f"No GPU found. Using CPU.")
        return torch.device("cpu")


def imageIdentifier(fetch: Fetch):
    print(f"Setting up the environment...")
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
                    print(f"Generating image caption...")
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
                    print(f"Generating answer...")
                    result = vqa_pipeline({"image": image, "question": question})
                    return result[0]['answer']

                else:
                    print(f"Invalid mode selected.")
                    pass

        except FileNotFoundError:
            print(f"File does not exist")
            pass


def imageGenerator(fetch: Fetch):
    print("Setting up the environment...")
    ig_pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if fetch._device.type == "cuda" else torch.float32
    )
    ig_pipeline.to(fetch._device)

    print(f"Image generating, this will take a while, please wait...")
    image = ig_pipeline(fetch._prompt).images[0]
    os.makedirs("generated_imgs", exist_ok=True)

    filename = os.path.join("generated_imgs", fileCounter(fetch))
    image.save(filename)
    print(f"Image saved as: {filename}")

def fileCounter(fetch: Fetch):
    os.makedirs("generated_imgs", exist_ok=True)

    while True:
        filename = f"generated_image{fetch._counter}.jpg"
        full_path = os.path.join("generated_imgs", filename)

        if os.path.exists(full_path):
            fetch._counter += 1
        else:
            return filename


if __name__ == "__main__":
    main()
