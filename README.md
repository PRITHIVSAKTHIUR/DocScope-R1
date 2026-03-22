# **DocScope-R1**

DocScope-R1 is an experimental, advanced document vision suite designed for high-performance Optical Character Recognition (OCR) and complex visual reasoning. Built on top of the robust Qwen2.5-VL architecture, this application provides a modern, interactive web interface that allows users to seamlessly upload documents, screenshots, receipts, or complex scene images. By integrating multiple state-of-the-art vision-language models, including Nvidia's Cosmos-Reason1-7B and specialized OCR variants, the tool empowers users to extract dense text, caption images, and perform deep visual analysis. The suite is fully GPU-accelerated and features granular control over text generation parameters, making it a highly versatile environment for testing and deploying vision-based artificial intelligence workflows.

<img width="1920" height="1798" alt="Screenshot 2026-03-22 at 11-54-05 DocScope-R1 - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/74fc8b7b-b9d6-4236-876e-a09e87e43d9c" />

### **Key Features**

* **Multi-Model Architecture:** Seamlessly switch between specialized vision-language models directly from the interface. Supported models include `Cosmos-Reason1-7B`, `docscopeOCR-7B`, `Captioner-7B-Qwen2.5VL`, and `visionOCR-3B`.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend built with custom HTML, CSS, and JavaScript. It includes a drag-and-drop media zone, real-time output streaming, and an integrated settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by adjusting parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated response as a `.txt` file.
* **Flash Attention 2 Integration:** Utilizes `kernels-community/flash-attn2` for optimized, memory-efficient inference on compatible GPUs.

### **Repository Structure**

```text
├── images/
│   ├── 1.jpg
│   └── 2.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run DocScope-R1 locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
torchvision
matplotlib
requests
kernels
hf_xet
spaces
pillow
gradio
av
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the models will be downloaded and loaded into VRAM upon their first invocation.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/DocScope-R1.git](https://github.com/PRITHIVSAKTHIUR/DocScope-R1.git)
