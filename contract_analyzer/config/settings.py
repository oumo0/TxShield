import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StorageConfig:
    json_dir: str = "contract_data_output"
    vector_dir: str = "vector_storage"
    results_dir: str = "analysis_results"
    cache_dir: str = ".cache"


@dataclass
class ModelConfig:
    provider: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    max_retries: int = 3
    timeout: int = 60
    enabled: bool = True


class ConfigManager:
    
    def __init__(self):
        self.storage = StorageConfig()
        self.http_proxy = os.getenv("HTTP_PROXY")
        self.https_proxy = os.getenv("HTTPS_PROXY")
        
        # AI model settings
        self.models = {
            "baidu": ModelConfig(
                provider="baidu",
                api_key=os.getenv("BAIDU_API_KEY"),
                api_url="url_to_baidu_service"
            ),
            "deepseek": ModelConfig(
                provider="deepseek",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_url="url_to_deepseek_service"
            ),
            "gitee": ModelConfig(
                provider="gitee",
                api_key=os.getenv("GITEE_API_KEY"),
                api_url="url_to_gitee_service"
            ),
            "local": ModelConfig(
                provider="local",
                api_url=os.getenv("LOCAL_API_URL", "http://localhost:8000"),
                enabled=os.getenv("LOCAL_ENABLED", "false").lower() == "true"
            )
        }
        
        # Embedding settings
        self.embedding_model = "embedding-model-name"
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.embedding_url = "url_to_embedding_service"
        
        # Performance settings
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "SIMILARITY_THRESHOLD"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "MAX_RETRIES"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "RETRY_DELAY"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "BATCH_SIZE"))
        self._setup_directories()
    
    def _setup_directories(self):
        dirs = [
            self.storage.json_dir,
            self.storage.vector_dir,
            self.storage.results_dir,
            self.storage.cache_dir
        ]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        return {name: config for name, config in self.models.items() 
                if config.enabled}
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        return self.models.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage": {
                "json_dir": self.storage.json_dir,
                "vector_dir": self.storage.vector_dir,
                "results_dir": self.storage.results_dir,
                "cache_dir": self.storage.cache_dir
            },
            "models": {
                name: {
                    "provider": config.provider,
                    "enabled": config.enabled,
                    "max_retries": config.max_retries,
                    "timeout": config.timeout
                }
                for name, config in self.models.items()
            },
            "performance": {
                "similarity_threshold": self.similarity_threshold,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "batch_size": self.batch_size
            }
        }


CONFIG = ConfigManager()
CONFIG_DICT = CONFIG.to_dict()