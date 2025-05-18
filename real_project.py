# Image Identifier or Image Generator Project
from transformers import pipeline
from PIL import Image
import sys
from diffusers import StableDiffusionPipeline
import torch


class Fetch:
    def __init__(self, prompt: str):
        self._prompt = prompt

    def imageIdentifier(self):
        print("Setting up the environment...")
        try:
            with Image.open(self._prompt) as img:
                image = img.convert("RGB")
                mode: int = int(input("Type '0' for captioning or '1' for Q&A: ").strip())

                if mode == 0:
                    # Captioning mode
                    caption_pipeline = pipeline(
                        "image-to-text",
                        model="Salesforce/blip-image-captioning-base"
                    )
                    print("Generating image caption...")
                    result = caption_pipeline(image)
                    return result[0]['generated_text']

                elif mode == 1:
                    # Visual Q&A mode
                    vqa_pipeline = pipeline(
                        "visual-question-answering",
                        model="Salesforce/blip-vqa-base",
                        tokenizer="Salesforce/blip-vqa-base"
                    )
                    question: str = input("Ask a question about the image: ")
                    print("Generating answer...")
                    result = vqa_pipeline({"image": image, "question": question})
                    return result[0]['answer']

                else:
                    return "Invalid mode selected."

        except FileNotFoundError:
            sys.exit("File does not exist")

    def imageGenerator(self):
        print("Setting up the environment...")
        ig_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32
        ).to("cpu")

        print("Image generating, this will take a while, please wait...")
        image = ig_pipeline(self._prompt).images[0]
        image.save("generated_image.png")

    @property
    def prompt(self):
        return self._prompt


def main():
    while True:
        try:
            task: int = int(input("Type '0' for Image Identification | Type '1' for Image Generation: \nInput: ").strip())

            if task == 0:
                prompt: str = input("Input filename (make sure that image is saved in the root folder): ").strip()
                if prompt.endswith("jpg") or prompt.endswith("png"):
                    answer = Fetch(prompt).imageIdentifier()
                    print(f"Result: {answer}")
                    break
                else:
                    print("Not a valid file type")
            elif task == 1:
                prompt: str = input("Generate a/an? ").strip()
                Fetch(prompt).imageGenerator()
                print("Done generating, locate the image in the root folder")
                break
            else:
                print("Not a valid input")
        except ValueError:
            print("Not a valid input")


if __name__ == "__main__":
    main()
