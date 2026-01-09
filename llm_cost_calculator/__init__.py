"""
LLM Cost Calculator - Comprehensive cost calculation and optimization for LLM APIs.

Features:
- Multi-provider cost comparison
- Request optimization recommendations
- Budget tracking and alerts
- Cost forecasting
- Token usage analytics
- Extended 2025 model pricing database
"""

from .calculator import (
    CostCalculator,
    ModelPricing,
    RequestCost,
    CostComparison,
    CostOptimization,
    BudgetTracker,
    CostForecast,
    BudgetAlert,
    Provider,
)
from .models import (
    EXTENDED_PRICING_DB,
    get_all_models,
    get_models_by_provider,
    get_models_by_quality,
    get_cheapest_model,
    get_fastest_model,
    get_best_value_model,
    find_model_by_capability,
    ModelSelector,
)

__version__ = "1.1.0"
__all__ = [
    # Core calculator
    "CostCalculator",
    "ModelPricing",
    "RequestCost",
    "CostComparison",
    "CostOptimization",
    "BudgetTracker",
    "BudgetAlert",
    "CostForecast",
    "Provider",
    # Extended models
    "EXTENDED_PRICING_DB",
    "get_all_models",
    "get_models_by_provider",
    "get_models_by_quality",
    "get_cheapest_model",
    "get_fastest_model",
    "get_best_value_model",
    "find_model_by_capability",
    "ModelSelector",
]
