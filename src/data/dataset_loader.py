import os
from datasets import load_dataset
from pathlib import Path

def download_dataset():
    """
    Tải Vietnamese Instruction Following dataset từ HuggingFace
    Sử dụng dataset 52K instruction mẫu của PhoGPT/Alpaca-vi hoặc tương đương.
    Ở đây sử dụng bộ 'Binh234/vietnamese-alpaca' làm baseline chuẩn.
    """
    print("Đang tải dataset Vietnamese Alpaca từ Hugging Face...")
    
    # Dataset vietnamese-alpaca là phiên dịch từ Stanford Alpaca 52K
    dataset_name = "bkai-foundation-models/vi-alpaca"
    
    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"Đã tải dataset '{dataset_name}' thành công. Số lượng samples: {len(dataset)}")
        
        # Tạo thư mục data nếu chưa có
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Lưu file thô để backup
        raw_path = data_dir / "raw_alpaca_vi.jsonl"
        dataset.to_json(raw_path, force_ascii=False)
        print(f"Đã lưu raw dataset tại: {raw_path}")
        
        return dataset
        
    except Exception as e:
        print(f"Lỗi tải dataset: {e}")
        return None

if __name__ == "__main__":
    download_dataset()
