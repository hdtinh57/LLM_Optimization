# Project 2: LLM Optimization & CoT Distillation Pipeline

An end-to-end LLM fine-tuning, knowledge distillation, and deployment pipeline explicitly optimized for consumer hardware (8GB VRAM RTX 3070). This project demonstrates how to distill reasoning capabilities (Chain-of-Thought) from large commercial models into a small, fast local model, and deploy it efficiently.

## 🚀 Key Features

- **Model**: Distilling into `Qwen/Qwen3.5-0.8B`
- **Knowledge Distillation**: Dual-provider CoT generator using **Gemini 2.5 Flash** and **Alibaba Cloud Qwen3.5-Flash** as teachers.
- **Method**: QLoRA (4-bit base + 16-bit adapters)
- **Framework**: [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and 70% VRAM reduction.
- **Dataset**: Custom synthesized Vietnamese CoT Dataset (derived from `bkai-foundation-models/vi-alpaca`).
- **Quantization**: Native GGUF export via `Q4_K_M` using Llama.cpp for highly optimized local inference.
- **Deployment**: FastAPI backend and native Ollama integration.

## 📈 Benchmark Results (Qwen 0.8B)

| Metrics         | Base Model (Qwen3.5-0.8B)                     | CoT Distilled Model                                                                                           |
| :-------------- | :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **Avg Speed**   | ~9.8 tokens/sec                               | ~7.4 tokens/sec                                                                                               |
| **Peak VRAM**   | 1.47 GB                                       | 1.48 GB                                                                                                       |
| **Qualitative** | Direct, sometimes generic, hallucinates logic | Explicitly outputs reasoning steps wrapped in `<think>...</think>` tags before answering. Coherent structure. |

## 📂 Project Structure

- `configs/`: Contains YAML configurations for standard training, quantization, and distillation.
- `src/`: Core logic (data processing, model configuration, `SFTTrainer` wrapper).
- `scripts/`: Sequential executable scripts for the complete pipeline.
- `data/`: Local dataset storage (ignored by git).
- `outputs/`: Saved model checkpoints and quantized GGUF binaries (ignored by git).
- `Modelfile`: Ollama deployment configuration for the distilled GGUF model.

## 🛠️ Usage Pipeline

Follow the sequential execution scripts to reproduce the pipeline from scratch:

**Phase 1 & 2: Data & Base Training**

1. `python scripts/01_setup_verify.py` - Verify CUDA & Torch compatibility.
2. `python scripts/02_prepare_data.py` - Download & process the base dataset.
3. `python scripts/03_train_qlora.py --epochs 3` - Run standard QLoRA fine-tuning.

**Phase 3: Quantization** 4. `python scripts/04_export_gguf.py` - Export trained model natively to GGUF format via Unsloth.

**Phase 4: CoT Knowledge Distillation** 5. `python scripts/05_generate_cot.py --provider dashscope --num-samples 1000` - Generate CoT instruction subset from Teacher AI. 6. `python scripts/06_train_cot_student.py --epochs 3` - Distill CoT capabilities into the student model.

**Phase 5 & 6: Deployment & Evaluation** 7. Build & Run locally with Ollama: `ollama create qwen-cot-0.8b -f Modelfile` 8. Start the FastAPI Inference Server: `python scripts/07_api_server.py` 9. Run formal Evaluation Benchmarks: `python scripts/08_evaluate.py`
