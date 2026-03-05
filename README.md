# Project 2: LLM Optimization Pipeline

This is an end-to-end LLM fine-tuning and deployment pipeline explicitly optimized for consumer hardware (8GB VRAM RTX 3070).

## 🚀 Key Features

- **Model**: Qwen/Qwen3.5-4B
- **Method**: QLoRA (4-bit base + 16-bit adapters)
- **Framework**: [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and 70% VRAM reduction.
- **Dataset**: Vietnamese Instructions (bkai-foundation-models/vi-alpaca)
- **Quantization**: GGUF export for local deployment
- **Deployment**: FastAPI & Ollama integration

## 📂 Project Structure

- `configs/`: Contains YAML configuration for training and distillation.
- `src/`: Core logic (data processing, model configuration).
- `scripts/`: Executable scripts for verification, training, and exporting.
- `data/`: Directory for local dataset storage (ignored by git).
- `outputs/`: Directory for saved model checkpoints (ignored by git).

## 🛠️ Usage

_For detailed instructions on setup and runtime, refer to the documentation and execution scripts in `/scripts`._
