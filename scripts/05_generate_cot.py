"""
Phase 4 - Task 4.1: Generate Chain-of-Thought (CoT) dataset using Teacher LLM API.
Supports two providers:
  - gemini  : Google Gemini API (google.genai SDK)
  - dashscope : Alibaba Cloud DashScope API (OpenAI-compatible, Qwen models)

Usage:
  python scripts/05_generate_cot.py --provider gemini --num-samples 100
  python scripts/05_generate_cot.py --provider dashscope --num-samples 100
  python scripts/05_generate_cot.py --provider dashscope --resume
"""
import os
import sys
import json
import time
import argparse
import random

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config import settings


# ============================================================
# Provider Abstraction
# ============================================================

class TeacherProvider:
    """Base class for teacher model providers."""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(TeacherProvider):
    """Google Gemini API provider."""
    def __init__(self, model_name: str):
        from google import genai
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"  Provider: Gemini | Model: {model_name}")

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip() if response.text else ""


class DashScopeProvider(TeacherProvider):
    """Alibaba Cloud DashScope API provider (OpenAI-compatible)."""
    def __init__(self, model_name: str):
        from openai import OpenAI
        api_key = settings.dashscope_api_key
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in .env file.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = model_name
        print(f"  Provider: DashScope (Alibaba Cloud) | Model: {model_name}")

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Ban la mot chuyen gia AI. Hay suy nghi tung buoc (Chain of Thought) roi dua ra cau tra loi."},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""


def create_provider(provider_name: str, model_override: str = None) -> TeacherProvider:
    """Factory function to create the appropriate provider."""
    if provider_name == "gemini":
        model = model_override or settings.distillation.teacher_model
        return GeminiProvider(model)
    elif provider_name == "dashscope":
        model = model_override or settings.distillation.dashscope_model
        return DashScopeProvider(model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use 'gemini' or 'dashscope'.")


# ============================================================
# Prompt & Format Helpers
# ============================================================

def build_cot_prompt(instruction: str, input_text: str = "") -> str:
    """Build the CoT prompt to send to the teacher model."""
    question = instruction
    if input_text:
        question = f"{instruction}\n\nDu lieu dau vao:\n{input_text}"

    prompt = (
        f"Cau hoi / Nhiem vu:\n{question}\n\n"
        "Hay tra loi theo cau truc sau:\n"
        "<think>\n"
        "[Qua trinh suy nghi tung buoc cua ban]\n"
        "</think>\n\n"
        "Cau tra loi cuoi cung:\n"
        "[Cau tra loi ngan gon, chinh xac]\n"
    )
    return prompt


def generate_with_retry(provider: TeacherProvider, prompt: str, max_retries: int = 3) -> str:
    """Call teacher API with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            return provider.generate(prompt)
        except Exception as e:
            wait_time = 2 ** attempt + random.random()
            print(f"  [Retry {attempt+1}/{max_retries}] Error: {e}. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    return ""


def format_training_sample(instruction: str, input_text: str, cot_response: str) -> dict:
    """Format a single sample into ChatML template for fine-tuning."""
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n{input_text}"

    text = (
        f"<|im_start|>system\n"
        f"Ban la mot tro ly AI thong minh. Hay suy nghi tung buoc truoc khi tra loi.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{cot_response}<|im_end|>"
    )
    return {"text": text}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate CoT dataset using Teacher LLM API")
    parser.add_argument("--provider", type=str, default="dashscope",
                        choices=["gemini", "dashscope"],
                        help="Teacher model provider (default: dashscope)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (default: from config)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help=f"Number of samples (default: {settings.distillation.num_samples})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output JSONL file (default: {settings.distillation.output_file})")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds (default: 1.0)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    num_samples = args.num_samples or settings.distillation.num_samples
    output_file = args.output or settings.distillation.output_file
    delay = args.delay

    print("=" * 60)
    print("Phase 4 - Task 4.1: Generate CoT Dataset")
    print("=" * 60)
    print(f"  Provider      : {args.provider}")
    print(f"  Num Samples   : {num_samples}")
    print(f"  Output File   : {output_file}")
    print(f"  API Delay     : {delay}s")
    print(f"  Resume Mode   : {args.resume}")

    # Step 1: Initialize provider
    print(f"\n[1/4] Initializing teacher provider...")
    provider = create_provider(args.provider, args.model)

    # Step 2: Load raw dataset
    raw_path = "data/raw_alpaca_vi.jsonl"
    print(f"\n[2/4] Loading raw dataset from {raw_path}...")
    samples = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"  Total raw samples: {len(samples)}")

    # Sample or limit
    if num_samples < len(samples):
        random.seed(42)
        samples = random.sample(samples, num_samples)
        print(f"  Randomly selected {num_samples} samples (seed=42)")

    # Step 3: Check resume
    existing_count = 0
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
        print(f"  Resuming from sample {existing_count}/{num_samples}")
        samples = samples[existing_count:]

    # Step 4: Generate CoT responses
    print(f"\n[3/4] Generating CoT responses...")
    success_count = existing_count
    fail_count = 0
    mode = "a" if args.resume and existing_count > 0 else "w"

    with open(output_file, mode, encoding="utf-8") as out_f:
        for i, sample in enumerate(samples):
            idx = existing_count + i + 1
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")

            # Build prompt and call API
            prompt = build_cot_prompt(instruction, input_text)
            cot_response = generate_with_retry(provider, prompt)

            if cot_response:
                training_sample = format_training_sample(instruction, input_text, cot_response)
                out_f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
                out_f.flush()
                success_count += 1
                status = "OK"
            else:
                fail_count += 1
                status = "FAIL"

            # Progress log
            if idx % 10 == 0 or idx == num_samples or idx <= 3:
                print(f"  [{idx}/{num_samples}] {status} | Success: {success_count} | Fail: {fail_count}")

            # Rate limiting
            time.sleep(delay)

    # Summary
    print(f"\n[4/4] Generation complete!")
    print("=" * 60)
    print(f"  Total Processed : {success_count + fail_count}")
    print(f"  Success         : {success_count}")
    print(f"  Failed          : {fail_count}")
    print(f"  Output          : {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
