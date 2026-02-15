"""BLIP-2 Visual QA Demo — Gradio app with 3 tabs."""

import traceback

import gradio as gr
from PIL import Image
from src.blip2_model import generate_caption, visual_qa, describe_image
from src.visualization import ensure_rgb


def _format_error(e: Exception) -> str:
    """Format an exception with its traceback for display in the UI."""
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    return f"[Error] {type(e).__name__}: {e}\n\n{''.join(tb)}"


def caption_tab(image):
    if image is None:
        return "Please upload an image."
    try:
        image = ensure_rgb(Image.fromarray(image))
        return generate_caption(image)
    except Exception as e:
        return _format_error(e)


def vqa_tab(image, question):
    if image is None:
        return "Please upload an image."
    if not question or not question.strip():
        return "Please enter a question."
    try:
        image = ensure_rgb(Image.fromarray(image))
        return visual_qa(image, question.strip())
    except Exception as e:
        return _format_error(e)


def compare_tab(image1, image2):
    if image1 is None or image2 is None:
        return "Please upload both images."
    try:
        img1 = ensure_rgb(Image.fromarray(image1))
        img2 = ensure_rgb(Image.fromarray(image2))

        desc1 = describe_image(img1)
        desc2 = describe_image(img2)

        result = f"**Image 1:** {desc1}\n\n**Image 2:** {desc2}\n\n"
        result += "**Differences:** The first image shows " + desc1.lower()
        result += ", while the second image shows " + desc2.lower() + "."
        return result
    except Exception as e:
        return _format_error(e)


with gr.Blocks(title="BLIP-2 Visual QA Demo") as demo:
    gr.Markdown("# 🖼️ BLIP-2 Visual QA Demo")
    gr.Markdown("Powered by [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)")

    with gr.Tab("Image Captioning"):
        with gr.Row():
            cap_input = gr.Image(label="Upload Image")
            cap_output = gr.Textbox(label="Caption", lines=3)
        cap_btn = gr.Button("Generate Caption", variant="primary")
        cap_btn.click(caption_tab, inputs=cap_input, outputs=cap_output)

    with gr.Tab("Visual QA"):
        with gr.Row():
            vqa_image = gr.Image(label="Upload Image")
            with gr.Column():
                vqa_question = gr.Textbox(label="Question", placeholder="What is in this image?")
                vqa_output = gr.Textbox(label="Answer", lines=3)
        vqa_btn = gr.Button("Ask", variant="primary")
        vqa_btn.click(vqa_tab, inputs=[vqa_image, vqa_question], outputs=vqa_output)

    with gr.Tab("Image Comparison"):
        with gr.Row():
            cmp_img1 = gr.Image(label="Image 1")
            cmp_img2 = gr.Image(label="Image 2")
        cmp_output = gr.Markdown(label="Comparison Result")
        cmp_btn = gr.Button("Compare", variant="primary")
        cmp_btn.click(compare_tab, inputs=[cmp_img1, cmp_img2], outputs=cmp_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
