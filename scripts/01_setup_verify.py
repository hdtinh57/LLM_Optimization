import os
import sys
import platform

def verify_system():
    print("="*50)
    print("System Information")
    print("="*50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    try:
        import torch
        print("\n" + "="*50)
        print("PyTorch & CUDA Verification")
        print("="*50)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available:  {'Yes' if torch.cuda.is_available() else 'No'}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version:    {torch.version.cuda}")
            print(f"Device Count:    {torch.cuda.device_count()}")
            print(f"Current Device:  {torch.cuda.current_device()}")
            print(f"Device Name:     {torch.cuda.get_device_name(0)}")
            
            # Khảo sát VRAM cho RTX 3070 (8GB)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total VRAM:      {vram_gb:.2f} GB")
            if vram_gb < 7.5:
                print("WARNING: VRAM < 8GB có thể gây lỗi OOM khi train Qwen3.5-4B.")
            else:
                print("OK: VRAM (>= 8GB) đủ cho QLoRA 4-bit config.")
        else:
            print("ERROR: Không nhận diện được CUDA. Hãy kiểm tra cài đặt PyTorch cu128.")

    except ImportError:
        print("ERROR: Chưa cài đặt PyTorch. Hãy chạy `pip install -r requirements.txt`")
        sys.exit(1)

def verify_configs():
    print("\n" + "="*50)
    print("Configuration Verification")
    print("="*50)
    try:
        # Thêm thư mục gốc vào path để import được thư mục src
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.config import settings
        
        print("OK: Load config thành công từ YAML và ENV.")
        print(f"Base Model:   {settings.base.model_id}")
        print(f"Dataset:      {settings.base.dataset_name}")
        print(f"Teacher:      {settings.distillation.teacher_model}")
        print(f"Target Mod:   {settings.base.target_modules}")
        
        if settings.gemini_api_key:
             print("GEMINI_API_KEY: Đã cấu hình")
        else:
             print("GEMINI_API_KEY: Chưa cấu hình (Cần thiết cho CoT Distillation)")
             
        if settings.wandb_api_key:
             print("WANDB_API_KEY:  Đã cấu hình")
        else:
             print("WANDB_API_KEY:  Chưa cấu hình (Khuyên dùng để tracking)")

    except Exception as e:
        print(f"ERROR: Load config thất bại. Chi tiết: {e}")

if __name__ == "__main__":
    verify_system()
    verify_configs()
    print("\nQuá trình verify kết thúc.")
