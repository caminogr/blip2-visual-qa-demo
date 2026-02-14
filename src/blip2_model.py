"""BLIP-2 model wrapper for captioning, VQA, and description."""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

MODEL_NAME = "Salesforce/blip2-opt-2.7b"

_processor = None
_model = None


def get_model():
    """Lazy-load the BLIP-2 model and processor."""
    global _processor, _model
    if _processor is None:
        print(f"Loading {MODEL_NAME}...")
        _processor = Blip2Processor.from_pretrained(MODEL_NAME)
        _model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        _model.eval()
        print("Model loaded.")
    return _processor, _model


def generate_caption(image: Image.Image) -> str:
    """Generate a caption for the given image."""
    processor, model = get_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption


def visual_qa(image: Image.Image, question: str) -> str:
    """Answer a question about the given image."""
    processor, model = get_model()
    inputs = processor(images=image, text=question, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    answer = processor.decode(output[0], skip_special_tokens=True).strip()
    return answer


def describe_image(image: Image.Image) -> str:
    """Generate a detailed description of the image."""
    processor, model = get_model()
    prompt = "Describe this image in detail:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150)
    description = processor.decode(output[0], skip_special_tokens=True).strip()
    return description
