import json
from pathlib import Path
from datasets import Dataset

def format_alpaca(example):
    """
    Format data dict thành Alpaca style text prompt.
    Tương thích với format prompt chuẩn cho Qwen instruction tuning.
    """
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output_text = example.get("output", "").strip()
    
    # Qwen chat template chuẩn thường sử dụng ChatML (<|im_start|>user...)
    # Nhưng với QLoRA cơ bản có thể dùng Alpaca format thô hoặc định dạng chuẩn.
    # Trong processor này, chúng ta lưu định dạng text thô để trainer/tokenizer tự quyết định template.
    
    if input_text:
         text = f"Dưới đây là một hướng dẫn mô tả một nhiệm vụ, đi kèm với đầu vào cung cấp thêm ngữ cảnh. Hãy viết phản hồi hoàn thành yêu cầu một cách thích hợp.\n\n### Hướng dẫn:\n{instruction}\n\n### Đầu vào:\n{input_text}\n\n### Phản hồi:\n{output_text}"
    else:
         text = f"Dưới đây là một hướng dẫn mô tả một nhiệm vụ. Hãy viết phản hồi hoàn thành yêu cầu một cách thích hợp.\n\n### Hướng dẫn:\n{instruction}\n\n### Phản hồi:\n{output_text}"
         
    example["text"] = text
    return example

def process_dataset(raw_dataset_path: str, output_path: str):
    """
    Đọc raw dataset, format theo chuẩn train, và lưu dict jsonl mới
    """
    print(f"Đang xử lý data từ {raw_dataset_path}...")
    
    path = Path(raw_dataset_path)
    if not path.exists():
        print(f"Lỗi: Không tìm thấy {raw_dataset_path}")
        return
        
    PROCESSED_DATA = []
    
    with open(path, "r", encoding="utf-8") as f:
         for line in f:
             if not line.strip(): continue
             try:
                 example = json.loads(line)
                 processed = format_alpaca(example)
                 PROCESSED_DATA.append({"text": processed["text"]})
             except json.JSONDecodeError:
                 continue
                 
    # Lưu lại file process
    out_path = Path(output_path)
    out_path.parent.mkdir(exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
         for item in PROCESSED_DATA:
             f.write(json.dumps(item, ensure_ascii=False) + "\n")
             
    print(f"Đã xử lý và lưu thành công {len(PROCESSED_DATA)} samples tại {output_path}")

if __name__ == "__main__":
    process_dataset("data/raw_alpaca_vi.jsonl", "data/processed_train.jsonl")
