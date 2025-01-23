# PaliGemma: Custom Vision-Language Model Implementation

## Overview
A custom implementation of PaliGemma, a multimodal vision-language model combining SigLIP vision encoder with Gemma language model.

## Features
- Custom multimodal transformer architecture
- SigLIP-based vision encoder
- Gemma language model integration
- Custom inference pipeline
- Image and text preprocessing

## Project Structure
- `inference.py`: Token generation and model inference
- `model_siglip.py`: Vision encoder implementation
- `modeling_gamma.py`: Core model architecture
- `processing_paligamma.py`: Image/text preprocessing
- `utils.py`: Model loading utilities

## Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU recommended

## Installation
1. Clone repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Inference
### Prepare Weights
1. Download weights from HuggingFace PaliGemma repository
2. Place in `./weights/` directory

### Running Inference
```bash
chmod +x launch_inference.sh
./launch_inference.sh
```

## Key Implementation Details
- Multimodal cross-attention mechanism
- Custom token generation with top-p sampling
- Flexible image preprocessing
- Support for different device types (CUDA, MPS, CPU)

## Limitations
- Single image per inference
- Experimental implementation
- Performance may vary from official implementation

## License
MIT License