# Image-Identifier-and-Generator-using-Python

#### Video Demo:  
<URL HERE>

---

#### Description:  
This project focuses on utilizing HuggingFace free AI models such as Salesforce-BLIP and CompVis-Stable-Diffusion by integrating it in an Image Identification and Image Generation Project. The project's goal is to implement CS50's course lectures like OOP, Error Handling, Unit Testing, Loops, Conditionals, etc.

---

### Project Directory Structure
cs50_project
│-- img/
│-- project.py
│-- test_project.py
│-- requirements.txt

---

### Package Requirements (`requirements.txt`)
- **Transformers & Diffusers** — These two packages are necessary for models to be locally used. Additionally, they provide a pipeline (easy) way of integrating models into the project.
- **accelerate** — This is a pre-requisite for using the pipelines of Transformers and Diffusers because some Hugging Face pipelines may throw warnings or fail when trying to use CUDA features.
- **Pillow** — This package includes the `Image` module which is extremely important for opening images.
- **Torch** — This package is also a pre-requisite of the Transformers and Diffusers packages. It handles all deep-learning operations. Without PyTorch, none of the models will run.
- **os** — This package is used for removing files or checking if a file exists in the directory. It is used specifically for this project.

---

### Main Functionalities
- **Image Identification** — Image captioning and Visual Question Answering are present, utilizing the `Salesforce/blip-image-captioning-base` and `Salesforce/blip-vqa-base` models, both found on the HuggingFace platform for free.
- **Image Generation** — Generates an image from a text prompt.

---

### Main Application (`project.py`)
- **Class: Fetch**  
  Encapsulates and initializes the user prompt (a file name for image identification or a description for image generation), device checker, and file counter.

- **Function: checkDevice()**  
  Detects whether to use CUDA (GPU, specifically NVIDIA GPUs) or CPU based on availability.

- **Function: fileCounter(fetch: Fetch)**  
  Prevents file overwriting. It uses conditional statements with Python’s built-in `os` package to check if a file exists. A `while` loop is used to increment the filename until a unique one is found.

- **Function: imageGenerator(fetch: Fetch)**  
  Utilizes the `checkDevice()` method to detect the available processing unit, then uses the `StableDiffusionPipeline` from the Diffusers package to generate an image based on the prompt. Finally, it saves the image using the `fileCounter()` method.

- **Function: imageIdentifier(fetch: Fetch)**  
  Offers two options:
  - `"0"` for captioning, where the model (`blip-image-captioning-base`) automatically describes the image.
  - `"1"` for Visual Question Answering, where the model (`blip-vqa-base`) answers a user-defined question about the image.

- **Function: main()**  
  The entry point of the project. It handles user interaction and calls either `imageGenerator()` or `imageIdentifier()` based on the selected task.

---

### Test Application (`test_project.py`)
- **Function: test_checkDevice()**  
  Calls the `checkDevice()` method and verifies if the device returns a valid output.

- **Function: test_fileCounter()**  
  Simulates an existing file and ensures that the `fileCounter()` method increments correctly and returns a valid output.

- **Function: test_imageGenerator(monkeypatch)**  
  Creates a dummy image using the Pillow package. Patches the pipeline, feeds a prompt to the `Fetch` class, runs the image generation method, and checks that the file is present in the directory.  
  Note: The classes inside this test replicate the important methods of `StableDiffusionPipeline`. This bypasses the need to load pre-trained models. Some variables may appear unused but are necessary for the test to function.

- **Function: test_imageIdentifier(monkeypatch)**  
  Similar to `test_imageGenerator()`, this test mocks a version of the pipeline but only tests the captioning part of `imageIdentifier()`.  
  It replaces the actual image file access with a dummy `Image.open()` implementation that behaves like a real context manager. It also mocks Hugging Face’s `pipeline()` function to return a predictable caption (not an image), allowing the test to verify that the captioning functionality correctly extracts and returns the text `"Caption"`.

---

### TODO
