import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread
from typing import Iterable

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2
import requests

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
    AutoModel,
    AutoTokenizer,
)
from transformers.image_utils import load_image
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# --- Theme and CSS Definition ---

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",  # SteelBlue base color
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Cosmos-Reason1-7B
MODEL_ID_M = "nvidia/Cosmos-Reason1-7B"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load DocScope
MODEL_ID_X = "prithivMLmods/docscopeOCR-7B-050425-exp"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Relaxed
MODEL_ID_Z = "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed"
processor_z = AutoProcessor.from_pretrained(MODEL_ID_Z, trust_remote_code=True)
model_z = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Z,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load visionOCR
MODEL_ID_V = "prithivMLmods/visionOCR-3B-061125"
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_V,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "Cosmos-Reason1-7B":
        processor, model = processor_m, model_m
    elif model_name == "docscopeOCR-7B-050425-exp":
        processor, model = processor_x, model_x
    elif model_name == "Captioner-7B-Qwen2.5VL":
        processor, model = processor_z, model_z
    elif model_name == "visionOCR-3B":
        processor, model = processor_v, model_v
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "Cosmos-Reason1-7B":
        processor, model = processor_m, model_m
    elif model_name == "docscopeOCR-7B-050425-exp":
        processor, model = processor_x, model_x
    elif model_name == "Captioner-7B-Qwen2.5VL":
        processor, model = processor_z, model_z
    elif model_name == "visionOCR-3B":
        processor, model = processor_v, model_v
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

# Define examples for image and video inference
image_examples = [
    ["Perform OCR on the text in the image.", "images/1.jpg"],
    ["Explain the scene in detail.", "images/2.jpg"]
]

video_examples = [
    ["Explain the Ad in Detail", "videos/1.mp4"],
    ["Identify the main actions in the video", "videos/2.mp4"]
]

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    gr.Markdown("# **DocScope R1**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Upload Image", height=290)
                    image_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=image_examples, inputs=[image_query, image_upload])
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Upload Video", height=290)
                    video_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=video_examples, inputs=[video_query, video_upload])
                    
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
                
        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            raw_output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=11, show_copy_button=True)
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown()
                    
            model_choice = gr.Radio(
                choices=["Cosmos-Reason1-7B", "docscopeOCR-7B-050425-exp", "Captioner-7B-Qwen2.5VL", "visionOCR-3B"],
                label="Select Model",
                value="Cosmos-Reason1-7B"
            )
            
    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[raw_output, markdown_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[raw_output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(mcp_server=True, ssr_mode=False, show_error=True)
