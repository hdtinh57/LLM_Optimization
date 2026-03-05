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

def export_to_gguf(quantization_methods: list):
    """
    Load fine-tuned model and export it to GGUF formats natively using Unsloth.
    Automatically merges LoRA weights into the base model before quantizing.
    """
    model_path = settings.base.output_dir
    output_dir = "outputs/quantized"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not Path(model_path).exists():
        print(f"Lỗi: Không tìm thấy model tại {model_path}. Vui lòng chạy 03_train_qlora.py trước.")
        sys.exit(1)

    print(f"Đang tải model (Base + LoRA) từ: {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = settings.base.max_seq_len,
        dtype = None,
        load_in_4bit = True, 
        local_files_only = True, # Bypass unsloth statistics/HF telemetry timeout
    )
    
    # Save the model in GGUF
    base_model_name = settings.base.model_id.split("/")[-1]

    for q_method in quantization_methods:
        print(f"\n[{q_method}] Đang xuất và Quantize model sang GGUF...")
        save_path = os.path.join(output_dir, f"{base_model_name}-unsloth")
        
        try:
            # Unsloth calls llama.cpp locally (downloads if necessary) to export the gguf file.
            model.save_pretrained_gguf(
                save_path, 
                tokenizer, 
                quantization_method = q_method
            )
            print(f"✅ Đã lưu GGUF ({q_method}) thành công.")
        except Exception as e:
            print(f"⚠️ Không thể xuất trực tiếp sang GGUF (Lỗi: {e}).")
            print("Chuyển sang phương án dự phòng: Lưu model merged 16-bit (safetensors).")
            # Fallback to saving merged 16-bit HF model. You can manually use llama.cpp to convert this later.
            fallback_path = os.path.join(output_dir, f"{base_model_name}-merged-16bit")
            model.save_pretrained_merged(fallback_path, tokenizer, save_method="merged_16bit")
            print(f"✅ Đã lưu fallback 16-bit model tại: {fallback_path}")
            print("Vui lòng tải llama.cpp binary cho Windows và chạy lồng để biên dịch sang GGUF thủ công.")
        
def main():
    parser = argparse.ArgumentParser(description="Export Unsloth model to GGUF using Llama.cpp")
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=[settings.quantization.gguf.qtype],
                        help="Tùy chọn danh sách format: f16, q4_k_m, q8_0, etc.")
    args = parser.parse_args()

    print("="*50)
    print("Khởi chạy Tiến trình GGUF Export (Unsloth + Llama.cpp)")
    print("="*50)
    
    export_to_gguf(args.methods)
    print("\nHoàn tất Export.")

if __name__ == "__main__":
    main()
