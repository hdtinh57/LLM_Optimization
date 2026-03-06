# Development Log - LLM Optimization Pipeline

> **⚠️ QUY ĐỊNH BẮT BUỘC:**
>
> 1. **TRƯỚC KHI BẮT ĐẦU CÔNG VIỆC MỚI:** Đọc toàn bộ file này để nắm context.
> 2. **SAU KHI HOÀN THÀNH CÔNG VIỆC:** Ghi log theo template bên dưới.
> 3. **KHI FIX BUG:** Ghi rõ triệu chứng → nguyên nhân → giải pháp.
> 4. **KHÔNG ĐƯỢC SKIP** việc ghi log dưới bất kỳ lý do nào.

---

## Log Template

```markdown
## YYYY-MM-DD

### [Phase X] Task X.X: Tên task

- **Trạng thái:** ✅ Hoàn thành / 🔄 Đang làm / ❌ Blocked
- **Công việc đã làm:**
  - Mô tả chi tiết...
- **Kết quả:**
  - Output cụ thể...
- **Ghi chú / Vấn đề:**
  - (nếu có)

### 🐛 Bug Fix: Tên bug

- **Triệu chứng:** Mô tả lỗi
- **Nguyên nhân gốc:** Root cause
- **Giải pháp:** Cách fix
- **File ảnh hưởng:** Danh sách file đã sửa
```

---

## 2026-03-04

### [Phase 0] Khởi tạo kế hoạch Project 2

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Đọc `AI_ML_Portfolio_Projects.md` — hiểu yêu cầu Project 2.
  - Tham khảo cấu trúc `Project1_SmartDocQA` (devlog, plan file, file structure).
  - Tạo bản kế hoạch chi tiết `llm-optimization-pipeline.md` với 6 Phase + Phase X.
  - Tạo file `devlog.md` này với quy định ghi log bắt buộc.
- **Kết quả:**
  - Plan file: `llm-optimization-pipeline.md` — roadmap đầy đủ.
  - Devlog: `devlog.md` — template và quy định.
  - Chờ user review và xác nhận các quyết định (model, dataset, teacher).
- **Ghi chú:**
  - Cần user chọn: Base model size (1.5B/3B/7B), Dataset, Teacher model.

### [Phase 0] User xác nhận quyết định + Research Qwen3.5-4B

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - User xác nhận 3 quyết định: **Qwen3.5-4B** / **Vietnamese Instruction Following** / **Gemini API**
  - Research Qwen3.5-4B trên HuggingFace:
    - 4B params, Hybrid Architecture (Gated DeltaNet + Gated Attention)
    - 262K native context, extensible to 1M tokens
    - Vision-Language model (Causal LM + Vision Encoder)
    - Thinking mode mặc định (`<think>...</think>`)
    - 201 ngôn ngữ, bao gồm tiếng Việt
  - Cập nhật plan file `llm-optimization-pipeline.md` với các quyết định và lưu ý kiến trúc.
- **Kết quả:**
  - Plan file updated, sẵn sàng chuyển sang Phase 1: Setup.
- **Ghi chú / Vấn đề tiềm ẩn:**
  - Hybrid Architecture (DeltaNet) có thể gây vấn đề tương thích với AWQ/GPTQ/GGUF — cần verify từng tool.
  - QLoRA `target_modules` cần tìm hiểu kỹ cho kiến trúc Gated DeltaNet + Gated Attention.

### [Phase 1] Task 1.1: Tạo Virtual Environment Python 3.13

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Đọc devlog (quy định bắt buộc) — nắm context Phase 0 đã xong.
  - Kiểm tra Python versions: `python` → 3.11.9, `py -3.13` → **3.13.12** ✅
  - Tạo venv: `py -3.13 -m venv venv` — thành công, exit code 0.
  - Verify: `.\venv\Scripts\python.exe --version` → **Python 3.13.12** ✅
- **Kết quả:**
  - Virtual environment tại `venv/` sử dụng Python 3.13.12.
- **Ghi chú:**

### [Phase 1] Task 1.2 & 1.3 & 1.4: Cấu hình Project & Cài Đặt Dependencies

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Tạo cấu trúc thư mục workflow chuẩn (`configs`, `src/data`, `outputs/models`...)
  - Phát sinh lỗi thiếu bản PyTorch `2.6.0+cu128` trên PyPI, đã bump up version lên `2.7.0+cu128`.
  - Loại bỏ các thư viện quantization (`autoawq`, `auto_gptq`, `llama-cpp-python`) khỏi `requirements.txt` do xung đột build C++ compiler trên Windows — sẽ cài đặt pre-built sau.
  - Cài đặt thành công toàn bộ Core LLM libraries, API & Utils (FastAPI, Wandb, Transformer, v.v.).
  - Tạo core pydantic-settings config tại `src/config.py` và các template YAML (`base_config.yaml`, v.v.).
  - Thiết lập verify script `scripts/01_setup_verify.py` hỗ trợ debug setup.
- **Kết quả:**
  - Lệnh verify xác nhận: PyTorch 2.7.0 kết nối mượt với CUDA 12.8, nhận diện RTX 3070, và cảnh báo cấu hình VRAM >= 8GB hợp lệ OK ✅.

### [Phase 2] Task 2.1: Chuẩn bị Dữ liệu (Vietnamese Instruction)

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Lập trình data loader `src/data/dataset_loader.py` và formater `src/data/data_processor.py`.
  - Xảy ra lỗi dataset template ban đầu `Binh234/vietnamese-alpaca` không tải được do source offline.
  - Đã đổi remote ID sang `bkai-foundation-models/vi-alpaca` (phổ biến hơn cho task này theo review trên mạng).
  - Khắc phục lỗi stdout unicode (Emoji) của command terminal trên Windows bằng ENV "PYTHONIOENCODING".
- **Kết quả:**
  - Tải và xử lý hoàn thiện `50006` sample Alpaca-style text. Lưu định dạng JSONL tại `data/processed_train.jsonl`.
- **Ghi chú:**
  - Database sẵn sàng cho khâu code train script. Cần báo cáo kết quả cho DEV và đợi confirmation cho Task 2.2.

### [Phase 2] Task 2.2 & 2.3: Lập trình và Khởi chạy QLoRA Trainer

- **Trạng thái:** ✅ Hoàn thành (Dry-run)
- **Công việc đã làm:**
  - Lập trình `src/models/trainer.py` khởi tạo Unsloth `FastLanguageModel` Load-in-4bit cùng với bộ TRL `SFTTrainer`.
  - Cấu hình PEFT model với r/alpha từ `configs/base_config.yaml` và áp dụng config cho các module Attention/MLP.
  - Fix config YAML pipeline: Đảm bảo class `BaseConfig` parse YAML dynamic khi initialize.
  - Giải quyết lỗi bộ nhớ/Window: Thay đổi model base từ `Qwen3.5-4B` xuống `Qwen3.5-0.8B` để tiết kiệm VRAM và giảm lỗi khi load optimizer, đồng thời disable Torch Dynamo/Triton qua env block do Triton chạy lỗi trên Windows (`$env:TORCHDYNAMO_DISABLE="1"`).
- **Kết quả:**
  - Script hoạt động trơn tru. Quá trình Dry Run (`--max-steps 1`) trên Model `Qwen3.5-0.8B` thành công!
  - Model weights LoRA (`qlora-qwen3.5-0.8b`) và file config đã được lưu đúng vị trí `outputs/models`. Thời gian train cho 1 step là ~28 giây.
- **Ghi chú:**
  - Đã sẵn sàng để User config số epoch đầy đủ và chạy train Full Data.
  - Sẽ thực hiện Merge code lên nhánh `dev` trên Git như kế hoạch ban đầu.

### [Phase 2] Task 2.1b: Switch pipeline sang thư viện Unsloth

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Nhận yêu cầu từ User thay thế cụm `autoawq, auto_gptq, llama-cpp-python` bằng `unsloth`.
  - Research Unsloth docs đối chiếu với Qwen3.5-4B. Phát hiện Unsloth hỗ trợ fine-tune và có sẵn hàm `save_pretrained_gguf`.
  - Kiểm tra script Unsloth trên máy Local nhưng gặp lỗi `CUDA: False` do pip tự động cài đè PyTorch CPU 2.10 khi cài Unsloth.
  - Sửa lỗi bằng cách ép cài đặt lại (force reinstall) `torch==2.7.0+cu128` qua pip. Unsloth đã nhận GPU (RTX 3070 8GB) thành công.
  - Cập nhật plan tổng `llm-optimization-pipeline.md`, `implementation_plan.md` và `task.md` lược bỏ các tools quantization cũ và gộp xuất GGUF thành option của Unsloth.
- **Kết quả:**
  - Môi trường hoàn chỉnh, tránh phụ thuộc rườm rà. Sẵn sàng train.

### [Phase 3] Task 3.1: Xuất mô hình sang định dạng GGUF

- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Viết script `scripts/04_export_gguf.py` gọi hàm `save_pretrained_gguf` built-in của Unsloth.
  - Tích hợp thành công **llama.cpp binary có sẵn của user (`llama-b8204-bin-win-cuda-12.4-x64`)**:
    - Clone source code llama.cpp nguyên bản từ git.
    - Copy toàn bộ executable `.exe` và `.dll` của bản pre-compiled mà user đã tải vào folder thư mục `llama.cpp` để đánh lừa Unsloth.
    - Unsloth gọi thành công script Python convert ra F16, và gọi tiếp `llama-quantize.exe` có sẵn để nén thành `Q4_K_M` mà không cần compile.
- **Kết quả:**
  - Lệnh export đã xuất thẳng sang GGUF Quantized mà không kích hoạt fallback 16-bit nữa.
  - File model lưu gốc tại: `outputs/quantized/Qwen3.5-0.8B-unsloth_gguf/Qwen3.5-0.8B.Q4_K_M.gguf`.
- **Ghi chú/Tiếp theo:**
  - Cập nhật lại thẻ `FROM` trong file `Modelfile` trỏ tới file `.Q4_K_M.gguf` chuẩn. User có thể tải model thẳng vào nền tảng Ollama với lệnh `ollama create qwen-cot-0.8b -f Modelfile`.

---

## 5. Phase 4: CoT Distillation

- **Mục tiêu:** Sinh dữ liệu suy luận (Chain-of-Thought) bằng Teacher Model API và train phân loại lên Student Model.
- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Code script sinh dataset `scripts/05_generate_cot.py` hỗ trợ dual-provider architecture: Google Gemini (`google.genai`) & Alibaba Cloud DashScope Qwen3.5-Flash (`openai`).
  - Hỗ trợ rate limiting, fallback exponential backoff, checkpoint resume.
  - Generat thành công 1000 mẫu CoT chất lượng cao kèm `<think>` tag.
  - Thiết lập và train Student: Viết `scripts/06_train_cot_student.py` tái chế Pipeline Unsloth SFTTrainer, truyền dataset nội bộ có kèm nhãn template ChatML.
  - Train thành công 3 Epoches, Loss giảm ngoạn mục từ `2.20` -> `1.13`.
  - Export Q4_K_M GGUF model mới, cập nhật `Modelfile`.

---

## 6. Phase 5: Deployment

- **Mục tiêu:** Deploy model local với Ollama layer qua FastAPI.
- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Viết server Inference `scripts/07_api_server.py` bằng FastAPI + Uvicorn + HTTPx.
  - Gọi Rest API cục bộ với Ollama local endpoint `v1/generate`.
  - Phơi bày `/health`, `/generate` và `/models`.
- **Kết quả:** Request streaming ổn định với tốc độ siêu việt ~133 tokens/s (cục bộ Q4_K_M).

---

## 7. Phase 6: Benchmark & Document

- **Mục tiêu:** So sánh sự khác rệt định lượng và định tính của Base vs CoT.
- **Trạng thái:** ✅ Hoàn thành
- **Công việc đã làm:**
  - Code `scripts/08_evaluate.py` dùng `AutoModelForCausalLM`, tải Base Qwen3.5-0.8B và Distilled Model tuần tự vào RAM.
  - Ràng buộc cấu hình `max_tokens` và quét Garbage Collection + giải phóng VRAM giữa phiên.
- **Kết quả:**
  - Model Base: Trả lời tự nhiên chung chung, 9.8 tokens/sec.
  - Model Distilled: Suy nghĩ bài bản qua tag `<think>`, theo logic đúng chuẩn nhưng dài hơn. Tốc độ hơi chậm lại do độ dài input/output (7.45 tokens/sec). VRAM nhỉnh hơn không đáng kể (+6MB).
  - Hoàn thiện file `README.md` với các Metric chi tiết và hướng dẫn sử dụng full 8 steps.
