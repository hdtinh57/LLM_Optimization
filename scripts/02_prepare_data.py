import os
import sys

# Thêm src vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import download_dataset
from src.data.data_processor import process_dataset

def main():
    print("🚀 Bắt đầu quá trình chuẩn bị dữ liệu...")
    
    # Bước 1: Download
    dataset = download_dataset()
    if dataset is None:
        print("🚨 Fetch dataset thất bại. Dừng script.")
        return
        
    # Bước 2: Format & Process
    process_dataset("data/raw_alpaca_vi.jsonl", "data/processed_train.jsonl")
    
    print("\n✅ Quá trình chuẩn bị dữ liệu kết thúc thành công.")

if __name__ == "__main__":
    main()
