# Image-Identifier-and-Generator-using-Python
#### Video Demo:  <URL HERE>
#### Description: This project focuses on utilizing HuggingFace free AI models such as Salesforce-Blip and CompVis-Stable-Diffusion by integrating it in an Image Identification and Image Generation Project. The project's goal is to implement cs50's course lectures like OOP, Error Handling, Unit Testing, Loops, Conditionals, and etc.

Project Directory Structure
cs50_project
    |--img/
    |-- project.py
    |-- test_project.py
    |-- requirements.txt


Package Requirements (requirements.txt)
    + Transformers & Diffusers -- this two package is necessary for models to be locally used. Additionally, it provides pipeline(easy) way of integrating it to the project.
    + accelerate -- this is a pre-requisite in using the pipelines of transformers and diffusers because some Hugging Face pipelines may throw warnings or fail when trying to use CUDA features.
    + Pillow -- this package includes Image module which is extremely needed for opening an image.
    + Torch -- this package is also a pre-requisite of transformer and diffusers package. It handles all deep-learning operations, without this pytorch, none of the models will run.
    + os -- this package is used in removing files or checking if file exists in the directory for this specific project.


Main Functionalities
    + Image Identification -- image captioning and Visual Question Answering are present utilizing the Salesforce\blip-Image-captioning-base and Salesforce\blip-vqa-base models and are both found in the HuggingFace platform for free.
    + Image Generation -- a text to image generation from a prompt
    

Main Application (project.py)
    + Class: Fetch -- this encapsulates and initializes the user prompt(a file name for image identification and a description for image generation), device checker, and file counter.
    + Function: checkDevice() -- this functionality detects whether to use CUDA (GPU specifically NVIDIA GPUs) or CPU based on availability.
    + Function: fileCounter(fetch: Fetch) -- this counter prevents the file replace existing files. This method utilizes conditional statement to check whether the file already exist or not, using the built-in package "os" in python. Additionally, for it to loop and counts succeedingly until the statement turns "False", while condition is utilized to achieve it.
    + Function: imageGeneration(fetch: Function) -- this method utilizes the checkDevice() method to confirm the device's processing unit availability and then utilizes the stablediffusionpipleine module from diffusers package to easily integrate it to the project. It then generates an image based on the prompt given to the project. Lastly, fileCounter() method is primarily used here in saving the image.
    + Function: imageIdentifier(fetch: Fetch) -- this method is divided into two other options, "0" for captioning where the model(blip-image-captioning-base) automatically identifies the image prompted from the main() function, while "1" shows another prompt for a specific command (eg. how many trees are there in the picture?, what is the object on the right called?, etc.) and the model(blip-qva-base) will generate an answer to it.
    + Function: main() -- this is the entry point of the whole project, where it is responsible for interacting with the user invoking the appropriate function (image generation or image identification).


Test Application (test_project.py)
    + Function: test_checkDevice() -- simply calls checkDevice() method from the main application and then assures if the device returns valid output.
    + Function: test_fileCounter() -- this simulates to exist a file and ensures if the fileCounter() method of the project icrements correctly and returns a valid output.
    + Function: test_imageGenerator(mokeypatch) -- this creates a dummy image using pillow package, pathes the pipeline, feeds a prompt to the class Fetch in the main application, runs the method that generates the image, and lastly, assures that the file is present in the directory. But something unusual int the code is the classes inside, the reason it is created is to replicate StableDiffusionPipeline important methods. This bypasses the need to load pre-trained models and even if some variables are not used, it is still necessarry to be able to run a test in the imageGenerator() method in the main application.
    + Function: test_imageIdentifier(mokeypatch) -- The same with test_imageGenerator() method, classes inside were present to mock a version of the pipeline and only one pair is seen because it only contains one test which will only test the captioning part of imageIdentification() method in the main application. What only differ this to the previous Test Function is it replaces the actual image file access with a dummy Image.open() implementation that behaves like a real context manager. It also mocks Hugging Face’s pipeline() function to return a predictable caption (not an image), allowing the test to verify that the captioning functionality correctly extracts and returns the text "Caption". 


    TODO
