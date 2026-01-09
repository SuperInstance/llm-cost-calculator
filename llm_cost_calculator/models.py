"""
Extended pricing database and model utilities with 2025 models.
"""

from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

from .calculator import (
    CostCalculator,
    ModelPricing,
    RequestCost,
    CostComparison,
    OptimizationRecommendation,
    BudgetTracker,
    BudgetAlert,
    CostForecast,
    Provider,
)


# Extended pricing database with additional 2025 models
EXTENDED_PRICING_DB: Dict[str, ModelPricing] = {
    # New OpenAI Models (2025)
    "gpt-4.1": ModelPricing(
        model_id="gpt-4.1",
        provider=Provider.OPENAI,
        input_price_per_million=2.00,
        output_price_per_million=8.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=350,
        quality_score=9.2
    ),
    "gpt-4.1-mini": ModelPricing(
        model_id="gpt-4.1-mini",
        provider=Provider.OPENAI,
        input_price_per_million=0.12,
        output_price_per_million=0.50,
        context_window=128000,
        max_output_tokens=16384,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=120,
        quality_score=7.8
    ),
    "gpt-4.1-turbo": ModelPricing(
        model_id="gpt-4.1-turbo",
        provider=Provider.OPENAI,
        input_price_per_million=5.00,
        output_price_per_million=20.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=300,
        quality_score=8.8
    ),
    "o3-mini": ModelPricing(
        model_id="o3-mini",
        provider=Provider.OPENAI,
        input_price_per_million=1.10,
        output_price_per_million=4.40,
        context_window=200000,
        max_output_tokens=100000,
        supports_function_calling=False,
        estimated_latency_ms=1500,
        quality_score=9.0
    ),
    "o3": ModelPricing(
        model_id="o3",
        provider=Provider.OPENAI,
        input_price_per_million=15.00,
        output_price_per_million=60.00,
        context_window=200000,
        max_output_tokens=100000,
        supports_function_calling=False,
        estimated_latency_ms=4000,
        quality_score=9.8
    ),

    # New Anthropic Models (2025)
    "claude-3.7-sonnet": ModelPricing(
        model_id="claude-3.7-sonnet",
        provider=Provider.ANTHROPIC,
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        context_window=200000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=500,
        quality_score=9.2
    ),
    "claude-3.5-sonnet-net": ModelPricing(
        model_id="claude-3.5-sonnet-net",
        provider=Provider.ANTHROPIC,
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        context_window=200000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=400,
        quality_score=9.0
    ),

    # New Google Models (2025)
    "gemini-2.0-flash": ModelPricing(
        model_id="gemini-2.0-flash",
        provider=Provider.GOOGLE,
        input_price_per_million=0.075,
        output_price_per_million=0.30,
        context_window=1000000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=100,
        quality_score=8.0
    ),
    "gemini-2.0-pro": ModelPricing(
        model_id="gemini-2.0-pro",
        provider=Provider.GOOGLE,
        input_price_per_million=1.25,
        output_price_per_million=5.00,
        context_window=2000000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=400,
        quality_score=8.8
    ),
    "gemini-2.5-flash": ModelPricing(
        model_id="gemini-2.5-flash",
        provider=Provider.GOOGLE,
        input_price_per_million=0.05,
        output_price_per_million=0.20,
        context_window=1000000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=80,
        quality_score=8.2
    ),
    "gemini-2.5-pro": ModelPricing(
        model_id="gemini-2.5-pro",
        provider=Provider.GOOGLE,
        input_price_per_million=1.00,
        output_price_per_million=4.00,
        context_window=2000000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=350,
        quality_score=9.0
    ),

    # New Meta Models (2025)
    "llama-3.3-70b-instruct": ModelPricing(
        model_id="llama-3.3-70b-instruct",
        provider=Provider.META,
        input_price_per_million=0.59,
        output_price_per_million=0.79,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=200,
        quality_score=8.2
    ),
    "llama-3.1-405b-instruct": ModelPricing(
        model_id="llama-3.1-405b-instruct",
        provider=Provider.META,
        input_price_per_million=2.70,
        output_price_per_million=2.70,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=400,
        quality_score=8.8
    ),
    "llama-3.2-90b": ModelPricing(
        model_id="llama-3.2-90b",
        provider=Provider.META,
        input_price_per_million=1.00,
        output_price_per_million=1.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=250,
        quality_score=8.3
    ),
    "llama-3.2-11b": ModelPricing(
        model_id="llama-3.2-11b",
        provider=Provider.META,
        input_price_per_million=0.15,
        output_price_per_million=0.15,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=80,
        quality_score=7.5
    ),
    "llama-3.2-3b": ModelPricing(
        model_id="llama-3.2-3b",
        provider=Provider.META,
        input_price_per_million=0.05,
        output_price_per_million=0.05,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=False,
        estimated_latency_ms=50,
        quality_score=6.5
    ),

    # New DeepSeek Models (2025)
    "deepseek-v3": ModelPricing(
        model_id="deepseek-v3",
        provider=Provider.DEEPSEEK,
        input_price_per_million=0.14,
        output_price_per_million=0.28,
        context_window=64000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=300,
        quality_score=8.5
    ),
    "deepseek-r1": ModelPricing(
        model_id="deepseek-r1",
        provider=Provider.DEEPSEEK,
        input_price_per_million=0.55,
        output_price_per_million=2.19,
        context_window=64000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=2000,
        quality_score=9.0
    ),
    "deepseek-r1-distill-llama-70b": ModelPricing(
        model_id="deepseek-r1-distill-llama-70b",
        provider=Provider.DEEPSEEK,
        input_price_per_million=0.73,
        output_price_per_million=0.73,
        context_window=32000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=800,
        quality_score=8.3
    ),
    "deepseek-r1-distill-qwen-32b": ModelPricing(
        model_id="deepseek-r1-distill-qwen-32b",
        provider=Provider.DEEPSEEK,
        input_price_per_million=0.36,
        output_price_per_million=0.36,
        context_window=32000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=500,
        quality_score=8.0
    ),

    # New Mistral Models (2025)
    "mistral-small-2501": ModelPricing(
        model_id="mistral-small-2501",
        provider=Provider.MISTRAL,
        input_price_per_million=0.10,
        output_price_per_million=0.10,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=100,
        quality_score=7.8
    ),
    "mistral-large-2501": ModelPricing(
        model_id="mistral-large-2501",
        provider=Provider.MISTRAL,
        input_price_per_million=2.00,
        output_price_per_million=6.00,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=400,
        quality_score=8.8
    ),
    "codestral-2501": ModelPricing(
        model_id="codestral-2501",
        provider=Provider.MISTRAL,
        input_price_per_million=0.20,
        output_price_per_million=0.20,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=150,
        quality_score=8.0
    ),
    "mistral-saba-2501": ModelPricing(
        model_id="mistral-saba-2501",
        provider=Provider.MISTRAL,
        input_price_per_million=0.50,
        output_price_per_million=0.50,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=300,
        quality_score=8.5
    ),

    # New Cohere Models (2025)
    "command-r7b": ModelPricing(
        model_id="command-r7b",
        provider=Provider.COHERE,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=100,
        quality_score=7.5
    ),
    "command-r-plus-08": ModelPricing(
        model_id="command-r-plus-08",
        provider=Provider.COHERE,
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=400,
        quality_score=8.8
    ),
    "command-a": ModelPricing(
        model_id="command-a",
        provider=Provider.COHERE,
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=350,
        quality_score=8.5
    ),

    # Qwen Models (Alibaba) - via Together
    "qwen-2.5-coder-32b-instruct": ModelPricing(
        model_id="qwen-2.5-coder-32b-instruct",
        provider=Provider.TOGETHER,
        input_price_per_million=0.30,
        output_price_per_million=0.30,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=200,
        quality_score=8.0
    ),
    "qwen-2.5-72b-instruct": ModelPricing(
        model_id="qwen-2.5-72b-instruct",
        provider=Provider.TOGETHER,
        input_price_per_million=0.90,
        output_price_per_million=0.90,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=350,
        quality_score=8.3
    ),
    "qwen-2.5-7b-instruct": ModelPricing(
        model_id="qwen-2.5-7b-instruct",
        provider=Provider.TOGETHER,
        input_price_per_million=0.08,
        output_price_per_million=0.08,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=80,
        quality_score=7.2
    ),

    # Groq-hosted models (very fast inference)
    "llama-3.3-70b-versatile-groq": ModelPricing(
        model_id="llama-3.3-70b-versatile-groq",
        provider=Provider.TOGETHER,
        input_price_per_million=0.59,
        output_price_per_million=0.79,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=50,  # Extremely fast on Groq
        quality_score=8.2
    ),
    "llama-3.1-8b-instant-groq": ModelPricing(
        model_id="llama-3.1-8b-instant-groq",
        provider=Provider.TOGETHER,
        input_price_per_million=0.10,
        output_price_per_million=0.10,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=30,  # Ultra fast
        quality_score=7.3
    ),
    "gemma2-9b-it-groq": ModelPricing(
        model_id="gemma2-9b-it-groq",
        provider=Provider.TOGETHER,
        input_price_per_million=0.10,
        output_price_per_million=0.10,
        context_window=128000,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=40,
        quality_score=7.5
    ),

    # xAI Models
    "grok-2": ModelPricing(
        model_id="grok-2",
        provider=Provider.ANYSCALE,
        input_price_per_million=2.00,
        output_price_per_million=10.00,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        supports_vision=True,
        estimated_latency_ms=500,
        quality_score=8.5
    ),
    "grok-beta": ModelPricing(
        model_id="grok-beta",
        provider=Provider.ANYSCALE,
        input_price_per_million=5.00,
        output_price_per_million=15.00,
        context_window=128000,
        max_output_tokens=8192,
        supports_function_calling=True,
        estimated_latency_ms=800,
        quality_score=9.0
    ),

    # Perplexity Models
    "sonar-pro": ModelPricing(
        model_id="sonar-pro",
        provider=Provider.ANYSCALE,
        input_price_per_million=1.00,
        output_price_per_million=1.00,
        context_window=127072,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=300,
        quality_score=8.3
    ),
    "sonar-small": ModelPricing(
        model_id="sonar-small",
        provider=Provider.ANYSCALE,
        input_price_per_million=0.20,
        output_price_per_million=0.20,
        context_window=127072,
        max_output_tokens=4096,
        supports_function_calling=True,
        estimated_latency_ms=150,
        quality_score=7.8
    ),
}


def get_all_models(include_extended: bool = True) -> Dict[str, ModelPricing]:
    """Get all models including extended pricing."""
    base = CostCalculator.PRICING_DB.copy()
    if include_extended:
        base.update(EXTENDED_PRICING_DB)
    return base


def get_models_by_provider(provider: Provider, include_extended: bool = True) -> Dict[str, ModelPricing]:
    """Get all models for a specific provider."""
    all_models = get_all_models(include_extended)
    return {k: v for k, v in all_models.items() if v.provider == provider}


def get_models_by_quality(min_quality: float, max_price: float = None, include_extended: bool = True) -> List[tuple[str, ModelPricing]]:
    """Get models filtered by quality and optional price cap."""
    all_models = get_all_models(include_extended)
    results = [(k, v) for k, v in all_models.items() if v.quality_score >= min_quality]

    if max_price is not None:
        # Filter by input price
        results = [(k, v) for k, v in results if v.input_price_per_million <= max_price]

    return sorted(results, key=lambda x: x[1].quality_score, reverse=True)


def get_cheapest_model(min_quality: float = 7.0, include_extended: bool = True) -> tuple[str, ModelPricing]:
    """Get the cheapest model above a quality threshold."""
    models = get_models_by_quality(min_quality, include_extended=include_extended)
    return min(models, key=lambda x: x[1].input_price_per_million + x[1].output_price_per_million)


def get_fastest_model(min_quality: float = 7.0, include_extended: bool = True) -> tuple[str, ModelPricing]:
    """Get the fastest model above a quality threshold."""
    models = get_models_by_quality(min_quality, include_extended=include_extended)
    return min(models, key=lambda x: x[1].estimated_latency_ms)


def get_best_value_model(min_quality: float = 7.0, include_extended: bool = True) -> tuple[str, ModelPricing]:
    """Get the best value model (quality/price ratio)."""
    models = get_models_by_quality(min_quality, include_extended=include_extended)
    return max(models, key=lambda x: x[1].quality_score / (x[1].input_price_per_million + x[1].output_price_per_million))


def find_model_by_capability(
    requires_function_calling: bool = False,
    requires_vision: bool = False,
    min_context_window: int = 0,
    max_price_per_million: Optional[float] = None,
    min_quality: float = 0.0,
    include_extended: bool = True
) -> List[tuple[str, ModelPricing]]:
    """Find models matching specific capabilities."""
    all_models = get_all_models(include_extended)
    results = []

    for model_id, pricing in all_models.items():
        if requires_function_calling and not pricing.supports_function_calling:
            continue
        if requires_vision and not pricing.supports_vision:
            continue
        if pricing.context_window < min_context_window:
            continue
        if pricing.quality_score < min_quality:
            continue
        if max_price_per_million and pricing.input_price_per_million > max_price_per_million:
            continue

        results.append((model_id, pricing))

    return sorted(results, key=lambda x: x[1].quality_score, reverse=True)


class ModelSelector:
    """
    Helper class for selecting models based on requirements.

    Example:
        selector = ModelSelector()
        model = selector.select(
            requirements={"quality": "high", "speed": "fast"},
            max_budget=0.50
        )
    """

    QUALITY_MAP = {
        "low": 6.5,
        "medium": 7.5,
        "high": 8.5,
        "premium": 9.0,
        "ultra": 9.5
    }

    SPEED_MAP = {
        "slow": 1000,  # > 1 second
        "medium": 500,
        "fast": 200,
        "ultra_fast": 100
    }

    def select(
        self,
        requirements: Dict[str, Any],
        max_budget: Optional[float] = None,
        provider: Optional[Provider] = None,
        include_extended: bool = True
    ) -> Optional[tuple[str, ModelPricing]]:
        """
        Select a model based on requirements.

        Args:
            requirements: Dict with keys like 'quality', 'speed', 'function_calling', 'vision'
            max_budget: Maximum price per million tokens
            provider: Filter by provider
            include_extended: Include extended pricing database

        Returns:
            Tuple of (model_id, ModelPricing) or None
        """
        min_quality = self.QUALITY_MAP.get(requirements.get("quality", "medium"), 7.5)
        max_latency = self.SPEED_MAP.get(requirements.get("speed", "medium"), 500)
        requires_fn = requirements.get("function_calling", False)
        requires_vision = requirements.get("vision", False)

        # Get candidate models
        candidates = find_model_by_capability(
            requires_function_calling=requires_fn,
            requires_vision=requires_vision,
            min_quality=min_quality,
            max_price_per_million=max_budget,
            include_extended=include_extended
        )

        # Filter by speed and provider
        filtered = []
        for model_id, pricing in candidates:
            if pricing.estimated_latency_ms > max_latency:
                continue
            if provider and pricing.provider != provider:
                continue
            filtered.append((model_id, pricing))

        if not filtered:
            return None

        # Return best match (prioritize quality, then price)
        return max(filtered, key=lambda x: (x[1].quality_score, -x[1].input_price_per_million))


__all__ = [
    # Extended database
    "EXTENDED_PRICING_DB",
    # Query functions
    "get_all_models",
    "get_models_by_provider",
    "get_models_by_quality",
    "get_cheapest_model",
    "get_fastest_model",
    "get_best_value_model",
    "find_model_by_capability",
    # Model selector
    "ModelSelector",
]
