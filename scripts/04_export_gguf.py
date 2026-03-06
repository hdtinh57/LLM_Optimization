import os
import sys
import argparse
from pathlib import Path

# Fix Windows/Triton Compatibility Issue for Unsloth
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from src.config import settings
import unsloth.models._utils

# Monkey-patch get_statistics to avoid timeout when HF Hub is unreachable
def dummy_get_statistics(*args, **kwargs):
    pass
unsloth.models._utils.get_statistics = dummy_get_statistics

def export_to_gguf(model_path: str, output_name: str, quantization_methods: list):
    """
    Load fine-tuned model and export it to GGUF formats natively using Unsloth.
    Automatically merges LoRA weights into the base model before quantizing.
    """
    output_dir = "outputs/quantized"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}.")
        sys.exit(1)

    print(f"Loading model (Base + LoRA) from: {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = settings.base.max_seq_len,
        dtype = None,
        load_in_4bit = True, 
        local_files_only = True,
    )
    
    for q_method in quantization_methods:
        print(f"\n[{q_method}] Exporting and quantizing to GGUF...")
        save_path = os.path.join(output_dir, f"{output_name}-unsloth")
        
        try:
            model.save_pretrained_gguf(
                save_path, 
                tokenizer, 
                quantization_method = q_method
            )
            print(f"OK: Saved GGUF ({q_method}) successfully.")
        except Exception as e:
            print(f"WARNING: Direct GGUF export failed ({e}).")
            print("Fallback: Saving merged 16-bit safetensors...")
            fallback_path = os.path.join(output_dir, f"{output_name}-merged-16bit")
            model.save_pretrained_merged(fallback_path, tokenizer, save_method="merged_16bit")
            print(f"OK: Saved fallback 16-bit model at: {fallback_path}")
        
def main():
    parser = argparse.ArgumentParser(description="Export Unsloth model to GGUF using Llama.cpp")
    parser.add_argument("--model-path", type=str, default=settings.base.output_dir,
                        help="Path to the trained model directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Output name prefix (default: derived from model_id)")
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=[settings.quantization.gguf.qtype],
                        help="Quantization methods: f16, q4_k_m, q8_0, etc.")
    args = parser.parse_args()

    output_name = args.name or settings.base.model_id.split("/")[-1]

    print("=" * 60)
    print("GGUF Export (Unsloth + Llama.cpp)")
    print("=" * 60)
    print(f"  Model Path : {args.model_path}")
    print(f"  Output Name: {output_name}")
    print(f"  Methods    : {args.methods}")
    print("=" * 60)
    
    export_to_gguf(args.model_path, output_name, args.methods)
    print("\nExport complete.")

if __name__ == "__main__":
    main()

