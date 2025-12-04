from typing import Dict, Any, Optional
from transformers import AutoConfig
from adapters import AdapterConfig, AutoAdapterModel
from .base_model import BaseModel

class AdapterBERT(BaseModel):
    def _setup_adapters(self):
        """Setup traditional adapters for the model."""
        # Add a new adapter
        adapter_name = "fake_news_detection"
        
        # Get reduction_factor with default value of 16 if not specified
        reduction_factor = self.adapter_config.get("reduction_factor", 16)
        
        # Create adapter configuration using the Pfeiffer architecture
        adapter_config = AdapterConfig.load(
            "pfeiffer",  # Traditional adapter architecture
            reduction_factor=reduction_factor,  # Bottleneck size
            non_linearity="relu",
        )
        
        # Add the adapter to the model
        self.model.add_adapter(adapter_name, config=adapter_config)
        
        # Activate the adapter
        self.model.set_active_adapters(adapter_name)
        
        # Train the adapter (freeze base model, train only adapter)
        self.model.train_adapter(adapter_name)
        
        # Freeze all parameters except adapter parameters
        for name, param in self.model.named_parameters():
            if 'adapter' not in name and 'LayerNorm' not in name:
                param.requires_grad = False
        
        print(f"Added adapter '{adapter_name}' with config: {adapter_config}")

    def forward(self, **inputs):
        """Forward pass with adapter."""
        # The adapters library handles everything automatically
        return self.model(**inputs)

    def save_pretrained(self, output_dir: str):
        """Save the model and its adapter configuration."""
        # Save the adapter and classification head
        self.model.save_adapter(
            output_dir,
            "fake_news_detection",
            with_head=True  # Save the classification head with the adapter
        )
        
        # Also save the full model for completeness
        self.model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load a pretrained model with adapters."""
        return cls(model_name=model_name, **kwargs)
