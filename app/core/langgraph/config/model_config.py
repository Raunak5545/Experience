"""Configuration for LLM models used in the workflow."""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MultimodalConfig:
    """Configuration specific to multimodal models."""
    generation_config: Optional[Dict] = None
    safety_settings: Optional[List[Dict]] = None
    debug_config: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    model_name: str
    temperature: float
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    multimodal: Optional[MultimodalConfig] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary, excluding None values."""
        config = {k: v for k, v in self.__dict__.items() if v is not None and k != 'multimodal'}
        return config


class WorkflowModelConfig:
    """Configuration for all models in the workflow."""

    def __init__(self):
        from app.core.config import settings

        # Default configuration that can be overridden
        self.extraction = ModelConfig(
            model_name=settings.EXTRACTION_MODEL,
            temperature=0.2,
            max_tokens=settings.MAX_TOKENS,
        )

        self.validation = ModelConfig(
            model_name=settings.VALIDATION_MODEL,
            temperature=0.2,
            max_tokens=settings.MAX_TOKENS,
        )

        self.classification = ModelConfig(
            model_name=settings.CLASSIFICATION_MODEL,
            temperature=0.1,
            max_tokens=settings.MAX_TOKENS,
        )

        self.basic_info = ModelConfig(
            model_name=settings.BASIC_INFO_MODEL,
            temperature=0.4,
            max_tokens=settings.MAX_TOKENS,
        )

        self.plan = ModelConfig(
            model_name=settings.PLAN_ITINERARY_MODEL,
            temperature=0.4,
            max_tokens=settings.MAX_TOKENS,
        )

        self.evaluation = ModelConfig(
            model_name=settings.EVALUATION_MODEL,
            temperature=0.2,
            max_tokens=settings.MAX_TOKENS,
        )

    def get_config(self, node_name: str) -> ModelConfig:
        """Get configuration for a specific node."""
        configs = {
            "extraction": self.extraction,
            "validation": self.validation,
            "classification": self.classification,
            "basic_info": self.basic_info,
            "plan": self.plan,
            "evaluation": self.evaluation,
        }
        return configs.get(node_name, self.extraction)  # Default to extraction config if not found


# Global instance for easy access
workflow_config = WorkflowModelConfig()