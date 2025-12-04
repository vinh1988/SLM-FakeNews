from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from adapters import AutoAdapterModel

class BaseModel(ABC, nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        adapter_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.adapter_config = adapter_config or {
            "adapter_size": 64,
            "adapter_initializer_range": 1e-4,
            "adapter_layer_norm_eps": 1e-12,
        }
        self.model = self._load_pretrained_model()
        self._setup_adapters()

    def _load_pretrained_model(self) -> PreTrainedModel:
        """Load the pretrained model."""
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        # Use AutoAdapterModel to load the model with adapter support
        model = AutoAdapterModel.from_pretrained(
            self.model_name,
            config=config,
        )
        # Add a classification head
        model.add_classification_head(
            "fake_news_detection",
            num_labels=self.num_labels
        )
        return model

    @abstractmethod
    def _setup_adapters(self):
        """Setup adapters for the model."""
        pass

    def forward(self, **inputs):
        return self.model(**inputs)

    def save_pretrained(self, output_dir: str):
        """Save the model and its configuration."""
        self.model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load a pretrained model."""
        return cls(model_name=model_name, **kwargs)
