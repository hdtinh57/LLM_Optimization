import os
import torch
from typing import Dict, Any
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

def setup_unsloth_model(model_name: str, max_seq_length: int, 
                        lora_r: int, lora_alpha: int, target_modules: list, 
                        lora_dropout: float) -> tuple:
    """
    Khởi tạo model từ Unsloth với FastLanguageModel và gắn PEFT (LoRA).
    Sử dụng bf16 để tối ưu hiệu năng và độ ổn định cho Qwen3.5.
    """
    print(f"Loading Unsloth model {model_name}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,           # Tự động chọn (bf16 cho Ampere/RTX 30xx)
        load_in_4bit = True,    # Sử dụng 4-bit config cho VRAM 8GB
    )

    print("Configuring PEFT/LoRA modules...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = target_modules,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Tối ưu hóa VRAM của Unsloth
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    return model, tokenizer

def create_trainer(model, tokenizer, dataset: Dataset, 
                   max_seq_length: int, output_dir: str, 
                   training_args: Dict[str, Any]) -> SFTTrainer:
    """
    Tạo TRL SFTTrainer kết hợp tham số từ Config.
    """
    print("Initializing SFTTrainer...")
    
    # Tạo cấu hình HF TrainingArguments
    args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = training_args.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 4),
        warmup_ratio = training_args.get("warmup_ratio", 0.03),
        num_train_epochs = training_args.get("num_train_epochs", 3),
        learning_rate = training_args.get("learning_rate", 2e-4),
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = training_args.get("logging_steps", 10),
        optim = training_args.get("optim", "adamw_8bit"),
        weight_decay = 0.01,
        lr_scheduler_type = training_args.get("lr_scheduler_type", "linear"),
        seed = 3407,
        save_strategy = "steps",
        save_steps = training_args.get("save_steps", 100),
        max_steps = training_args.get("max_steps", -1),
        report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"]
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Packing có thể tăng performance cho short inputs
        args = args,
    )

    return trainer
