import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from typing import Dict, Any, List, Optional
from pathlib import Path


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseYamlConfig(BaseSettings):
    """Base class to load default values from a YAML file."""
    yaml_file: str

    @model_validator(mode="before")
    @classmethod
    def load_from_yaml(cls, data: Any) -> Any:
        yaml_path = data.get("yaml_file")
        if yaml_path is None and "yaml_file" in cls.model_fields:
            yaml_path = cls.model_fields["yaml_file"].default

        if yaml_path and os.path.exists(yaml_path):
            yaml_data = load_yaml(yaml_path)
            # Update data with yaml data, but respect environment variables if passed
            for k, v in yaml_data.items():
                if k not in data:
                    data[k] = v
        return data


class BaseConfig(BaseYamlConfig):
    yaml_file: str = "configs/base_config.yaml"
    model_id: str = "Qwen/Qwen3.5-4B"
    output_dir: str = "outputs/models/qlora-qwen3.5-4b"
    dataset_name: str = "TIGER-Lab/MathInstruct"
    max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    num_train_epochs: int = 3
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    bf16: bool = False


class QuantizationAwqConfig(BaseSettings):
    w_bit: int = 4
    q_group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"


class QuantizationGptqConfig(BaseSettings):
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True


class QuantizationGgufConfig(BaseSettings):
    qtype: str = "q4_k_m"


class QuantizationConfig(BaseYamlConfig):
    yaml_file: str = "configs/quantization_config.yaml"
    awq: QuantizationAwqConfig = Field(default_factory=QuantizationAwqConfig)
    gptq: QuantizationGptqConfig = Field(default_factory=QuantizationGptqConfig)
    gguf: QuantizationGgufConfig = Field(default_factory=QuantizationGgufConfig)
    
    @model_validator(mode="before")
    @classmethod
    def load_from_yaml(cls, data: Any) -> Any:
         # Need custom parsing for nested objects
         yaml_path = data.get("yaml_file", "configs/quantization_config.yaml")
         if os.path.exists(yaml_path):
             yaml_data = load_yaml(yaml_path)
             if "awq" in yaml_data: data["awq"] = yaml_data["awq"]
             if "gptq" in yaml_data: data["gptq"] = yaml_data["gptq"]
             if "gguf" in yaml_data: data["gguf"] = yaml_data["gguf"]
         return data


class DistillationConfig(BaseYamlConfig):
    yaml_file: str = "configs/distillation_config.yaml"
    teacher_model: str = "gemini-2.5-flash"
    prompt_template: str = ""
    num_samples: int = 1000
    output_file: str = "data/cot_dataset.jsonl"


class Settings(BaseSettings):
    """Main Settings class aggregating all configurations and environment variables."""
    # Env vars
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    wandb_api_key: Optional[str] = Field(default=None, alias="WANDB_API_KEY")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    
    # Sub-configs
    base: BaseConfig = Field(default_factory=BaseConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global settings instance
settings = Settings()

if __name__ == "__main__":
    print(f"Loaded config: Model ID = {settings.base.model_id}")
    print(f"HF Token Set: {settings.hf_token is not None}")
