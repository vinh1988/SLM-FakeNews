import re
from typing import Optional, Dict, Any
import pandas as pd


class DataCleaner:
    """
    Data cleaning utilities for different fake news datasets.
    Handles text preprocessing and cleaning for WELFake, FakeNewsNet, and other datasets.
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize the data cleaner for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset ('welfake', 'fakenewsnet', 'fake_news')
        """
        self.dataset_name = dataset_name.lower()
    
    @staticmethod
    def clean_text_welfake(text: Any) -> str:
        """
        Clean text for WELFake dataset.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def clean_text_fakenewsnet(text: Any) -> str:
        """
        Clean text for FakeNewsNet dataset.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def clean_text_fakenews(text: Any) -> str:
        """
        Clean text for Fake News dataset.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs (both http and www)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text(self, text: Any) -> str:
        """
        Clean text based on the dataset type.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if self.dataset_name == 'welfake':
            return self.clean_text_welfake(text)
        elif self.dataset_name == 'fakenewsnet':
            return self.clean_text_fakenewsnet(text)
        elif self.dataset_name == 'fake_news':
            return self.clean_text_fakenews(text)
        else:
            # Default cleaning
            return self.clean_text_welfake(text)
    
    def combine_title_text(self, title: Any, text: Any, separator: str = " [SEP] ") -> str:
        """
        Combine title and text with a separator (useful for WELFake).
        
        Args:
            title: Title text
            text: Body text
            separator: Separator between title and text
            
        Returns:
            Combined text string
        """
        # Ensure both are strings
        title_str = title if isinstance(title, str) else ""
        text_str = text if isinstance(text, str) else ""
        
        # Handle empty strings
        if not title_str and not text_str:
            return ""
        elif not title_str:
            return text_str
        elif not text_str:
            return title_str
        
        return title_str + separator + text_str
    
    def clean_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single example from the dataset.
        
        Args:
            example: Dictionary containing the example data
            
        Returns:
            Cleaned example dictionary
        """
        if self.dataset_name == 'welfake':
            # Combine title and text, then clean
            title = example.get('title', '')
            text = example.get('text', '')
            combined = self.combine_title_text(title, text)
            example['content'] = self.clean_text(combined)
            
        elif self.dataset_name == 'fakenewsnet':
            # FakeNewsNet uses 'title' as the main text column
            title = example.get('title', '')
            example['content'] = self.clean_text(title)
            
        elif self.dataset_name == 'liar':
            # LIAR uses 'statement' as the text column
            statement = example.get('statement', '')
            example['content'] = self.clean_text(statement)
        
        return example
    
    def clean_batch(self, examples: Dict[str, list]) -> Dict[str, list]:
        """
        Clean a batch of examples from the dataset.
        
        Args:
            examples: Dictionary containing lists of example data
            
        Returns:
            Cleaned examples dictionary
        """
        if self.dataset_name == 'welfake':
            # Combine title and text for all examples
            titles = examples.get('title', [''] * len(examples.get('text', [])))
            texts = examples.get('text', [''] * len(examples.get('title', [])))
            
            # Combine and clean
            combined = [self.combine_title_text(t, txt) for t, txt in zip(titles, texts)]
            examples['content'] = [self.clean_text(c) for c in combined]
            
        elif self.dataset_name == 'fakenewsnet':
            # FakeNewsNet uses 'title' as the main text column
            titles = examples.get('title', [])
            examples['content'] = [self.clean_text(t) for t in titles]
            
        elif self.dataset_name == 'liar':
            # LIAR uses 'statement' as the text column
            statements = examples.get('statement', [])
            examples['content'] = [self.clean_text(s) for s in statements]
        
        return examples
    
    def __repr__(self) -> str:
        return f"DataCleaner(dataset_name='{self.dataset_name}')"

