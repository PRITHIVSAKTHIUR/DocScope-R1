# DocScope-R1

A powerful multi-modal AI application that combines three state-of-the-art vision-language models for comprehensive image and video analysis. DocScope-R1 provides OCR capabilities, detailed scene understanding, and video content analysis through an intuitive Gradio interface.

## Features

- **Multi-Model Support**: Choose from three specialized models for different tasks
- **Image Analysis**: Upload images for OCR, scene description, and detailed captioning
- **Video Processing**: Analyze videos with frame-by-frame understanding
- **Real-time Streaming**: Get responses as they are generated
- **Advanced Controls**: Fine-tune generation parameters for optimal results

## Supported Models

### 1. Cosmos-Reason1-7B (NVIDIA)
- **Purpose**: Physical common sense understanding and embodied decision making
- **Best for**: Reasoning about physical interactions and spatial relationships
- **Model**: `nvidia/Cosmos-Reason1-7B`

### 2. DocScope OCR-7B
- **Purpose**: Document-level optical character recognition
- **Best for**: Text extraction from documents and long-context vision-language understanding
- **Model**: `prithivMLmods/docscopeOCR-7B-050425-exp`

### 3. Captioner-Relaxed-7B
- **Purpose**: Detailed image captioning and description
- **Best for**: Generating comprehensive descriptions for text-to-image training data
- **Model**: `Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed`

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- 20GB+ free disk space for models

### Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install gradio
pip install spaces
pip install opencv-python
pip install pillow
pip install numpy
```

### Clone Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/DocScope-R1.git
cd DocScope-R1
```

## Usage

### Running the Application

```bash
python app.py
```

The application will start and provide you with a local URL (typically `http://127.0.0.1:7860`) to access the web interface.

### Image Analysis

1. Select the "Image Inference" tab
2. Enter your query in the text box
3. Upload an image
4. Choose your preferred model
5. Adjust advanced parameters if needed
6. Click "Submit"

**Example Queries:**
- "Perform OCR on the text in the image"
- "Explain the scene in detail"
- "Describe all objects and their relationships"

### Video Analysis

1. Select the "Video Inference" tab
2. Enter your query describing what you want to analyze
3. Upload a video file
4. Select the appropriate model
5. Configure generation parameters
6. Click "Submit"

**Example Queries:**
- "Explain the advertisement in detail"
- "Identify the main actions in the video"
- "Describe the sequence of events"

## Configuration

### Advanced Parameters

- **Max New Tokens** (1-2048): Maximum length of generated response
- **Temperature** (0.1-4.0): Controls randomness in generation
- **Top-p** (0.05-1.0): Nucleus sampling parameter
- **Top-k** (1-1000): Limits vocabulary for each step
- **Repetition Penalty** (1.0-2.0): Reduces repetitive outputs

### Environment Variables

- `MAX_INPUT_TOKEN_LENGTH`: Maximum input context length (default: 4096)

## Technical Details

### Video Processing

Videos are automatically downsampled to 10 evenly spaced frames for analysis. Each frame is processed with its timestamp and combined into a comprehensive understanding of the video content.

### Model Architecture

All models are based on the Qwen2.5-VL architecture with different fine-tuning objectives:
- Half-precision (float16) inference for efficiency
- GPU acceleration with CUDA support
- Streaming text generation for real-time responses

### Performance Optimization

- Models are loaded once at startup
- GPU memory is efficiently managed
- Streaming responses provide immediate feedback
- Automatic device detection (CUDA/CPU)

## File Structure

```
DocScope-R1/
├── app.py              # Main application file
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── images/            # Example images
│   ├── 1.jpg
│   └── 2.jpg
└── videos/            # Example videos
    ├── 1.mp4
    └── 2.mp4
```

## System Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070 or equivalent)
- **RAM**: 16GB system memory
- **Storage**: 25GB free space
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Recommended Requirements
- **GPU**: 12GB+ VRAM (RTX 4070 Ti or better)
- **RAM**: 32GB system memory
- **Storage**: SSD with 50GB free space
- **CPU**: High-performance processor (Intel i7/AMD Ryzen 7 or better)

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce max_new_tokens
- Lower the input resolution
- Use CPU inference (slower but works with limited VRAM)

**Model Loading Errors**
- Ensure stable internet connection for initial model download
- Check available disk space
- Verify Hugging Face access for gated models

**Video Processing Issues**
- Ensure video format is supported (MP4, AVI, MOV)
- Check video file isn't corrupted
- Reduce video length for large files

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- NVIDIA for the Cosmos-Reason1-7B model
- Qwen team for the base architecture
- Hugging Face for the transformers library
- Gradio team for the interface framework

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainer.
