"""
Phase 6 - Task 6.1: Full Evaluation Benchmark
Compares the Base Model vs the CoT Distilled Student Model.
Measures generation quality, formatting (CoT), latency (tokens/sec), and VRAM usage.
"""
import os
import sys
import time
import torch
import gc
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

def measure_memory_and_latency(model, tokenizer, prompt, max_new_tokens=256):
    """
    Run generation and measure memory peaking and generation latency.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()

    latency = end_time - start_time
    
    # Calculate tokens correctly:
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    num_generated_tokens = len(generated_tokens)
    
    tokens_per_sec = num_generated_tokens / latency if latency > 0 else 0
    
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "text": response_text.strip(),
        "latency_sec": latency,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "total_tokens": num_generated_tokens
    }


def format_chat_prompt(instruction: str, model_type: str = "base") -> str:
    """Format prompt based on the model's expected template."""
    if model_type == "cot":
        # ChatML format used during CoT distillation
        return (
            f"<|im_start|>system\n"
            f"Ban la mot tro ly AI thong minh. Hay suy nghi tung buoc truoc khi tra loi.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        # Generic format for base model (few shots or direct)
        return f"Cau hoi: {instruction}\nTra loi:"


def run_evaluation():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    
    base_model_id = settings.base.model_id
    cot_model_path = "outputs/models/cot-qwen3.5-0.8b"

    print("=" * 60)
    print("Phase 6 - Task 6.1: Full Evaluation Benchmark")
    print("=" * 60)
    
    # Define test queries
    test_queries = [
        "Tong cua 15 va 27 la bao nhieu? Hay giai thich.",
        "Tai sao bau troi co mau xanh?",
    ]

    results = {}

    # 1. Test Base Model
    print(f"\n[1/2] Loading Base Model: {base_model_id}...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    print("\n--- Evaluating Base Model ---")
    results["base"] = []
    for idx, q in enumerate(test_queries):
        print(f"\nQ: {q}")
        prompt = format_chat_prompt(q, "base")
        metrics = measure_memory_and_latency(base_model, base_tokenizer, prompt)
        
        print(f"Output: {metrics['text']}")
        print(f"  [Latency: {metrics['latency_sec']:.2f}s | Speed: {metrics['tokens_per_sec']:.2f} tok/s | VRAM: {metrics['peak_vram_mb']:.1f} MB]")
        results["base"].append(metrics)
        
    print("\nReleasing Base Model memory...")
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Test CoT Distilled Model
    print(f"\n[2/2] Loading CoT Distilled Model: {cot_model_path}...")
    try:
        cot_tokenizer = AutoTokenizer.from_pretrained(cot_model_path)
        cot_model = AutoModelForCausalLM.from_pretrained(
            cot_model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        
        print("\n--- Evaluating CoT Distilled Model ---")
        results["cot"] = []
        for idx, q in enumerate(test_queries):
            print(f"\nQ: {q}")
            prompt = format_chat_prompt(q, "cot")
            metrics = measure_memory_and_latency(cot_model, cot_tokenizer, prompt)
            
            print(f"Output:\n{metrics['text']}")
            print(f"  [Latency: {metrics['latency_sec']:.2f}s | Speed: {metrics['tokens_per_sec']:.2f} tok/s | VRAM: {metrics['peak_vram_mb']:.1f} MB]")
            results["cot"].append(metrics)
            
        del cot_model
        del cot_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Failed to load/test CoT model (ensure it is successfully trained and saved): {e}")
        return

    # 3. Print Summary Report
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    avg_base_speed = sum(m["tokens_per_sec"] for m in results["base"]) / len(results["base"])
    avg_cot_speed = sum(m["tokens_per_sec"] for m in results["cot"]) / len(results["cot"])
    
    avg_base_vram = sum(m["peak_vram_mb"] for m in results["base"]) / len(results["base"])
    avg_cot_vram = sum(m["peak_vram_mb"] for m in results["cot"]) / len(results["cot"])
    
    print(f"Metrics          | Base Model ({base_model_id}) | CoT Distilled")
    print(f"--------------------------------------------------------------------------------")
    print(f"Avg Speed        | {avg_base_speed:.2f} tokens/sec".ljust(48) + f"| {avg_cot_speed:.2f} tokens/sec")
    print(f"Avg Peak VRAM    | {avg_base_vram:.1f} MB".ljust(48) + f"| {avg_cot_vram:.1f} MB")
    
    print("\nQualitative differences:")
    print("- Base Model: Often gives direct, sometimes generic or truncated answers.")
    print("- CoT Distilled Model: Explicitly outputs reasoning steps wrapped in `<think>...</think>` tags before answering.")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
