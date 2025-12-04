from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class DatasetConfig:
    name: str
    hf_id: str
    text_column: str
    label_column: str
    test_size: float = 0.125
    val_size: float = 0.125
    # Class weights: (weight_for_class_0, weight_for_class_1)
    # Higher weight = more importance for that class
    class_weights: Tuple[float, float] = (1.0, 1.0)
    
# Dataset configurations
# Class weights calculated as: 1 / (2 * class_proportion)
# This balances the loss contribution from each class
DATASETS = {
    "welfake": DatasetConfig(
        name="welfake",
        hf_id="davanstrien/WELFake",
        text_column="text",
        label_column="label",
        class_weights=(0.92, 1.10)  # 54.5% Fake, 45.5% Real
    ),
    "liar": DatasetConfig(
        name="liar",
        hf_id="chengxuphd/liar2",
        text_column="statement",
        label_column="label",
        class_weights=(1.27, 0.83)  # 39.4% Fake, 60.6% Real
    ),
    "fakenewsnet": DatasetConfig(
        name="fakenewsnet",
        hf_id="rickstello/FakeNewsNet",
        text_column="title",
        label_column="real",
        class_weights=(2.07, 0.66)  # 24.2% Fake, 75.8% Real
    )
}

# Model configurations
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "minilm": "microsoft/MiniLM-L12-H384-uncased",
    "albert": "albert-base-v2"
}

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    return DATASETS[dataset_name.lower()]

def get_model_name(model_key: str) -> str:
    """Get the Hugging Face model name."""
    return MODELS[model_key.lower()]

def get_class_weights(dataset_name: str) -> Tuple[float, float]:
    """Get class weights for a specific dataset."""
    return DATASETS[dataset_name.lower()].class_weights
