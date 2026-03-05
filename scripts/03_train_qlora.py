import os
import sys
import argparse
from pathlib import Path
from datasets import load_dataset

# Fix Windows/Triton Compatibility Issue for Unsloth
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.models.trainer import setup_unsloth_model, create_trainer

def format_prompt(example):
    """
    Format logic: Since data is already processed into complete 'text' column 
    by src/data/data_processor.py, Unsloth SFTTrainer can use dataset_text_field="text".
    But if the 'text' field is present, we don't need further formatting.
    """
    return example

def main():
    parser = argparse.ArgumentParser(description="Run QLoRA Fine-tuning with Unsloth")
    parser.add_argument("--max-steps", type=int, default=None, 
                        help="Giới hạn số step train (dành cho chế độ dry-run/debug)")
    args = parser.parse_args()

    print("="*50)
    print("Khởi chạy Tiến trình Huấn luyện QLoRA (Unsloth)")
    print("="*50)

    # 1. Load Dataset
    data_path = "data/processed_train.jsonl"
    if not Path(data_path).exists():
        print(f"Lỗi: Không tìm thấy dữ liệu tại {data_path}. Vui lòng chạy 02_prepare_data.py trước.")
        sys.exit(1)
        
    print(f"Đọc dữ liệu từ {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Số lượng samples: {len(dataset)}")

    # 2. Setup Unsloth Model & Tokenizer
    model, tokenizer = setup_unsloth_model(
        model_name = settings.base.model_id,
        max_seq_length = settings.base.max_seq_len,
        lora_r = settings.base.lora_r,
        lora_alpha = settings.base.lora_alpha,
        target_modules = settings.base.target_modules,
        lora_dropout = settings.base.lora_dropout
    )

    # 3. Training Parameters Setup
    training_args = {
        "per_device_train_batch_size": settings.base.per_device_train_batch_size,
        "gradient_accumulation_steps": settings.base.gradient_accumulation_steps,
        "warmup_ratio": settings.base.warmup_ratio,
        "num_train_epochs": settings.base.num_train_epochs,
        "learning_rate": settings.base.learning_rate,
        "logging_steps": settings.base.logging_steps,
        "optim": settings.base.optim,
        "lr_scheduler_type": settings.base.lr_scheduler_type,
        "save_steps": settings.base.save_steps,
    }
    
    if args.max_steps:
        print(f"[DEBUG MODE] Giới hạn huẩn luyện ở {args.max_steps} steps.")
        training_args["max_steps"] = args.max_steps
        
    # 4. Initialize Trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_seq_length=settings.base.max_seq_len,
        output_dir=settings.base.output_dir,
        training_args=training_args
    )

    # 5. Start Training
    print("Bắt đầu huấn luyện...")
    trainer_stats = trainer.train()
    
    # 6. Save Model
    print("Huấn luyện hoàn tất. Đang lưu mô hình...")
    # Unsloth hỗ trợ lưu model 16bit và lora adapters
    model.save_pretrained(settings.base.output_dir)
    tokenizer.save_pretrained(settings.base.output_dir)
    print(f"Mô hình đã được lưu tại thư mục: {settings.base.output_dir}")
    
    # In thông kê kết quả
    print(f"Thời gian train: {trainer_stats.metrics['train_runtime']} giây")
    print(f"Loss trung bình: {trainer_stats.metrics['train_loss']}")

if __name__ == "__main__":
    main()
