"""
Core cost calculation and optimization functionality.
"""

import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict
import json


class Provider(str, Enum):
    """LLM Providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    TOGETHER = "together"
    ANYSCALE = "anyscale"


@dataclass
class ModelPricing:
    """Pricing information for a model"""
    model_id: str
    provider: Provider
    input_price_per_million: float  # USD
    output_price_per_million: float  # USD
    context_window: int
    max_output_tokens: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    estimated_latency_ms: int = 500
    quality_score: float = 8.0  # 1-10 subjective quality

    @property
    def input_price_per_token(self) -> float:
        return self.input_price_per_million / 1_000_000

    @property
    def output_price_per_token(self) -> float:
        return self.output_price_per_million / 1_000_000


@dataclass
class RequestCost:
    """Cost breakdown for a request"""
    model_id: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: float = field(default_factory=lambda: time.time())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_per_1k_tokens(self) -> float:
        if self.total_tokens == 0:
            return 0
        return (self.total_cost / self.total_tokens) * 1000


@dataclass
class CostComparison:
    """Result of comparing costs across models"""
    request_description: str
    input_tokens: int
    output_tokens: int
    comparisons: List[RequestCost]
    cheapest: RequestCost
    most_expensive: RequestCost
    savings_vs_cheapest: float
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class OptimizationRecommendation:
    """Recommendation for cost optimization"""
    current_model: str
    recommended_model: str
    reason: str
    estimated_savings_percent: float
    estimated_savings_usd: float
    quality_impact: str  # "minimal", "moderate", "significant"
    confidence: float  # 0-1


@dataclass
class BudgetAlert:
    """Alert when budget threshold is exceeded"""
    budget_name: str
    threshold_type: str  # "percent", "absolute"
    threshold_value: float
    current_spent: float
    current_percent: float
    message: str
    timestamp: float = field(default_factory=lambda: time.time())


class BudgetTracker:
    """Track spending against budgets"""

    def __init__(self):
        self.budgets: Dict[str, float] = {}
        self.spent: Dict[str, float] = defaultdict(float)
        self.alerts: List[BudgetAlert] = []

    def set_budget(self, name: str, amount_usd: float) -> None:
        """Set a budget"""
        self.budgets[name] = amount_usd

    def record_cost(self, budget_name: str, cost: float) -> Optional[BudgetAlert]:
        """Record a cost and check for alerts"""
        self.spent[budget_name] += cost
        return self._check_alerts(budget_name)

    def get_remaining(self, budget_name: str) -> float:
        """Get remaining budget"""
        if budget_name not in self.budgets:
            return 0
        return self.budgets[budget_name] - self.spent[budget_name]

    def get_percent_used(self, budget_name: str) -> float:
        """Get percentage of budget used"""
        if budget_name not in self.budgets or self.budgets[budget_name] == 0:
            return 0
        return (self.spent[budget_name] / self.budgets[budget_name]) * 100

    def _check_alerts(self, budget_name: str) -> Optional[BudgetAlert]:
        """Check if budget thresholds exceeded"""
        percent = self.get_percent_used(budget_name)
        spent = self.spent[budget_name]

        # Check 50% threshold
        if 50 <= percent < 51:
            return BudgetAlert(
                budget_name=budget_name,
                threshold_type="percent",
                threshold_value=50,
                current_spent=spent,
                current_percent=percent,
                message=f"Budget '{budget_name}' is 50% used (${spent:.2f} of ${self.budgets[budget_name]:.2f})"
            )

        # Check 75% threshold
        if 75 <= percent < 76:
            return BudgetAlert(
                budget_name=budget_name,
                threshold_type="percent",
                threshold_value=75,
                current_spent=spent,
                current_percent=percent,
                message=f"Budget '{budget_name}' is 75% used (${spent:.2f} of ${self.budgets[budget_name]:.2f})"
            )

        # Check 90% threshold
        if 90 <= percent < 91:
            return BudgetAlert(
                budget_name=budget_name,
                threshold_type="percent",
                threshold_value=90,
                current_spent=spent,
                current_percent=percent,
                message=f"⚠️ Budget '{budget_name}' is 90% used! Only ${self.get_remaining(budget_name):.2f} remaining."
            )

        # Check 100% exceeded
        if percent >= 100:
            return BudgetAlert(
                budget_name=budget_name,
                threshold_type="percent",
                threshold_value=100,
                current_spent=spent,
                current_percent=percent,
                message=f"🚨 Budget '{budget_name}' EXCEEDED by ${percent - 100:.1f}%!"
            )

        return None

    def get_status(self, budget_name: str) -> Dict[str, Any]:
        """Get budget status"""
        return {
            "budget": self.budgets.get(budget_name, 0),
            "spent": self.spent[budget_name],
            "remaining": self.get_remaining(budget_name),
            "percent_used": self.get_percent_used(budget_name),
            "alerts": [a for a in self.alerts if a.budget_name == budget_name]
        }


class CostForecast:
    """Forecast costs based on usage patterns"""

    def __init__(self):
        self.history: List[RequestCost] = []

    def add_request(self, cost: RequestCost) -> None:
        """Add a request to history"""
        self.history.append(cost)

    def forecast_daily(self, model_id: str, requests_per_day: int) -> float:
        """Forecast daily cost for a model"""
        model_costs = [c for c in self.history if c.model_id == model_id]
        if not model_costs:
            return 0

        avg_cost = sum(c.total_cost for c in model_costs) / len(model_costs)
        return avg_cost * requests_per_day

    def forecast_monthly(self, model_id: str, requests_per_day: int) -> float:
        """Forecast monthly cost (30 days)"""
        daily = self.forecast_daily(model_id, requests_per_day)
        return daily * 30

    def forecast_tokens_per_month(self, model_id: str, requests_per_day: int) -> Dict[str, int]:
        """Forecast token usage per month"""
        model_costs = [c for c in self.history if c.model_id == model_id]
        if not model_costs:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        avg_input = sum(c.input_tokens for c in model_costs) / len(model_costs)
        avg_output = sum(c.output_tokens for c in model_costs) / len(model_costs)

        monthly_requests = requests_per_day * 30

        return {
            "input_tokens": int(avg_input * monthly_requests),
            "output_tokens": int(avg_output * monthly_requests),
            "total_tokens": int((avg_input + avg_output) * monthly_requests)
        }


class CostCalculator:
    """
    Calculate and optimize LLM API costs.

    Example:
        calculator = CostCalculator()

        # Calculate single request cost
        cost = calculator.calculate(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500
        )

        # Compare across models
        comparison = calculator.compare(
            input_tokens=1000,
            output_tokens=500
        )

        # Get optimization recommendations
        recommendations = calculator.optimize(
            current_model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            requirements={"quality": "high"}
        )
    """

    # Pricing database (as of 2025)
    PRICING_DB: Dict[str, ModelPricing] = {
        # OpenAI
        "gpt-4o": ModelPricing(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            input_price_per_million=2.50,
            output_price_per_million=10.00,
            context_window=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=400,
            quality_score=9.0
        ),
        "gpt-4o-mini": ModelPricing(
            model_id="gpt-4o-mini",
            provider=Provider.OPENAI,
            input_price_per_million=0.15,
            output_price_per_million=0.60,
            context_window=128000,
            max_output_tokens=16384,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=150,
            quality_score=7.5
        ),
        "gpt-4-turbo": ModelPricing(
            model_id="gpt-4-turbo",
            provider=Provider.OPENAI,
            input_price_per_million=10.00,
            output_price_per_million=30.00,
            context_window=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=500,
            quality_score=8.5
        ),
        "gpt-4": ModelPricing(
            model_id="gpt-4",
            provider=Provider.OPENAI,
            input_price_per_million=30.00,
            output_price_per_million=60.00,
            context_window=8192,
            max_output_tokens=4096,
            supports_function_calling=True,
            estimated_latency_ms=800,
            quality_score=9.0
        ),
        "gpt-3.5-turbo": ModelPricing(
            model_id="gpt-3.5-turbo",
            provider=Provider.OPENAI,
            input_price_per_million=0.50,
            output_price_per_million=1.50,
            context_window=16385,
            max_output_tokens=4096,
            supports_function_calling=True,
            estimated_latency_ms=200,
            quality_score=7.0
        ),
        "o1-preview": ModelPricing(
            model_id="o1-preview",
            provider=Provider.OPENAI,
            input_price_per_million=15.00,
            output_price_per_million=60.00,
            context_window=128000,
            max_output_tokens=32768,
            supports_function_calling=False,
            estimated_latency_ms=5000,
            quality_score=9.5
        ),
        "o1-mini": ModelPricing(
            model_id="o1-mini",
            provider=Provider.OPENAI,
            input_price_per_million=1.10,
            output_price_per_million=4.40,
            context_window=128000,
            max_output_tokens=65536,
            supports_function_calling=False,
            estimated_latency_ms=2000,
            quality_score=8.5
        ),

        # Anthropic
        "claude-3-7-sonnet": ModelPricing(
            model_id="claude-3-7-sonnet",
            provider=Provider.ANTHROPIC,
            input_price_per_million=3.00,
            output_price_per_million=15.00,
            context_window=200000,
            max_output_tokens=8192,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=600,
            quality_score=9.0
        ),
        "claude-3-5-sonnet": ModelPricing(
            model_id="claude-3-5-sonnet",
            provider=Provider.ANTHROPIC,
            input_price_per_million=3.00,
            output_price_per_million=15.00,
            context_window=200000,
            max_output_tokens=8192,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=500,
            quality_score=9.0
        ),
        "claude-3-5-haiku": ModelPricing(
            model_id="claude-3-5-haiku",
            provider=Provider.ANTHROPIC,
            input_price_per_million=0.80,
            output_price_per_million=4.00,
            context_window=200000,
            max_output_tokens=8192,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=150,
            quality_score=8.0
        ),
        "claude-3-opus": ModelPricing(
            model_id="claude-3-opus",
            provider=Provider.ANTHROPIC,
            input_price_per_million=15.00,
            output_price_per_million=75.00,
            context_window=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=800,
            quality_score=9.5
        ),
        "claude-3-sonnet": ModelPricing(
            model_id="claude-3-sonnet",
            provider=Provider.ANTHROPIC,
            input_price_per_million=3.00,
            output_price_per_million=15.00,
            context_window=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=400,
            quality_score=8.5
        ),
        "claude-3-haiku": ModelPricing(
            model_id="claude-3-haiku",
            provider=Provider.ANTHROPIC,
            input_price_per_million=0.25,
            output_price_per_million=1.25,
            context_window=200000,
            max_output_tokens=4096,
            estimated_latency_ms=100,
            quality_score=7.5
        ),

        # Google
        "gemini-2.0-flash-exp": ModelPricing(
            model_id="gemini-2.0-flash-exp",
            provider=Provider.GOOGLE,
            input_price_per_million=0.075,
            output_price_per_million=0.30,
            context_window=1000000,
            max_output_tokens=8192,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=200,
            quality_score=8.0
        ),
        "gemini-1.5-pro": ModelPricing(
            model_id="gemini-1.5-pro",
            provider=Provider.GOOGLE,
            input_price_per_million=1.25,
            output_price_per_million=5.00,
            context_window=2000000,
            max_output_tokens=8192,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=500,
            quality_score=8.5
        ),
        "gemini-1.5-flash": ModelPricing(
            model_id="gemini-1.5-flash",
            provider=Provider.GOOGLE,
            input_price_per_million=0.075,
            output_price_per_million=0.30,
            context_window=1000000,
            max_output_tokens=8192,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=150,
            quality_score=7.5
        ),

        # Meta (via various providers)
        "llama-3.3-70b": ModelPricing(
            model_id="llama-3.3-70b",
            provider=Provider.META,
            input_price_per_million=0.59,
            output_price_per_million=0.79,
            context_window=128000,
            max_output_tokens=4096,
            estimated_latency_ms=300,
            quality_score=7.5
        ),
        "llama-3.1-405b": ModelPricing(
            model_id="llama-3.1-405b",
            provider=Provider.META,
            input_price_per_million=2.70,
            output_price_per_million=2.70,
            context_window=128000,
            max_output_tokens=4096,
            estimated_latency_ms=800,
            quality_score=8.5
        ),
        "llama-3.1-70b": ModelPricing(
            model_id="llama-3.1-70b",
            provider=Provider.META,
            input_price_per_million=0.59,
            output_price_per_million=0.79,
            context_window=128000,
            max_output_tokens=4096,
            estimated_latency_ms=300,
            quality_score=7.5
        ),
        "llama-3-70b": ModelPricing(
            model_id="llama-3-70b",
            provider=Provider.META,
            input_price_per_million=0.70,
            output_price_per_million=0.70,
            context_window=8192,
            max_output_tokens=4096,
            estimated_latency_ms=350,
            quality_score=7.0
        ),
        "llama-3-8b": ModelPricing(
            model_id="llama-3-8b",
            provider=Provider.META,
            input_price_per_million=0.10,
            output_price_per_million=0.10,
            context_window=8192,
            max_output_tokens=4096,
            estimated_latency_ms=100,
            quality_score=6.5
        ),

        # Mistral
        "mistral-large": ModelPricing(
            model_id="mistral-large",
            provider=Provider.MISTRAL,
            input_price_per_million=2.00,
            output_price_per_million=6.00,
            context_window=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            estimated_latency_ms=400,
            quality_score=8.5
        ),
        "mixtral-8x22b": ModelPricing(
            model_id="mixtral-8x22b",
            provider=Provider.MISTRAL,
            input_price_per_million=0.65,
            output_price_per_million=0.65,
            context_window=65536,
            max_output_tokens=4096,
            estimated_latency_ms=500,
            quality_score=8.0
        ),
        "mixtral-8x7b": ModelPricing(
            model_id="mixtral-8x7b",
            provider=Provider.MISTRAL,
            input_price_per_million=0.50,
            output_price_per_million=0.50,
            context_window=32768,
            max_output_tokens=4096,
            estimated_latency_ms=400,
            quality_score=7.5
        ),
        "codestral": ModelPricing(
            model_id="codestral",
            provider=Provider.MISTRAL,
            input_price_per_million=0.20,
            output_price_per_million=0.20,
            context_window=32000,
            max_output_tokens=4096,
            estimated_latency_ms=200,
            quality_score=7.0
        ),

        # DeepSeek
        "deepseek-chat": ModelPricing(
            model_id="deepseek-chat",
            provider=Provider.DEEPSEEK,
            input_price_per_million=0.14,
            output_price_per_million=0.28,
            context_window=128000,
            max_output_tokens=4096,
            estimated_latency_ms=400,
            quality_score=7.5
        ),
        "deepseek-coder": ModelPricing(
            model_id="deepseek-coder",
            provider=Provider.DEEPSEEK,
            input_price_per_million=0.14,
            output_price_per_million=0.28,
            context_window=128000,
            max_output_tokens=4096,
            estimated_latency_ms=400,
            quality_score=7.5
        ),
        "deepseek-reasoner": ModelPricing(
            model_id="deepseek-reasoner",
            provider=Provider.DEEPSEEK,
            input_price_per_million=0.55,
            output_price_per_million=2.19,
            context_window=64000,
            max_output_tokens=8192,
            estimated_latency_ms=3000,
            quality_score=8.5
        ),
    }

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        """
        Initialize the cost calculator.

        Args:
            custom_pricing: Optional custom pricing to add/override defaults
        """
        self.pricing = self.PRICING_DB.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)
        self.budget_tracker = BudgetTracker()
        self.forecast = CostForecast()

    def calculate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        record: bool = False
    ) -> RequestCost:
        """
        Calculate cost for a request.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            record: Whether to record in forecast history

        Returns:
            RequestCost with breakdown
        """
        pricing = self._get_pricing(model)
        provider = pricing.provider

        input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_million

        cost = RequestCost(
            model_id=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(input_cost + output_cost, 6)
        )

        if record:
            self.forecast.add_request(cost)

        return cost

    def compare(
        self,
        input_tokens: int,
        output_tokens: int,
        models: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> CostComparison:
        """
        Compare costs across models.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            models: Optional list of models to compare (default: all)
            filters: Optional filters (supports_vision, supports_function_calling, etc.)

        Returns:
            CostComparison with all comparisons
        """
        if models is None:
            models_to_check = list(self.pricing.keys())
        else:
            models_to_check = [m for m in models if m in self.pricing]

        # Apply filters
        if filters:
            if filters.get("supports_vision"):
                models_to_check = [m for m in models_to_check
                                   if self.pricing[m].supports_vision]
            if filters.get("supports_function_calling"):
                models_to_check = [m for m in models_to_check
                                   if self.pricing[m].supports_function_calling]
            if filters.get("min_quality"):
                min_q = filters["min_quality"]
                models_to_check = [m for m in models_to_check
                                   if self.pricing[m].quality_score >= min_q]
            if filters.get("max_latency_ms"):
                max_lat = filters["max_latency_ms"]
                models_to_check = [m for m in models_to_check
                                   if self.pricing[m].estimated_latency_ms <= max_lat]

        comparisons = [
            self.calculate(model, input_tokens, output_tokens)
            for model in models_to_check
        ]

        sorted_by_cost = sorted(comparisons, key=lambda c: c.total_cost)

        return CostComparison(
            request_description=f"{input_tokens} input + {output_tokens} output tokens",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            comparisons=comparisons,
            cheapest=sorted_by_cost[0],
            most_expensive=sorted_by_cost[-1],
            savings_vs_cheapest=sorted_by_cost[-1].total_cost - sorted_by_cost[0].total_cost
        )

    def optimize(
        self,
        current_model: str,
        input_tokens: int,
        output_tokens: int,
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Get cost optimization recommendations.

        Args:
            current_model: Current model being used
            input_tokens: Typical input tokens
            output_tokens: Typical output tokens
            requirements: Optional requirements (quality, features, etc.)

        Returns:
            List of optimization recommendations
        """
        requirements = requirements or {}
        current_cost = self.calculate(current_model, input_tokens, output_tokens)

        # Get comparison with filters based on requirements
        filters = {}
        if requirements.get("needs_vision"):
            filters["supports_vision"] = True
        if requirements.get("needs_function_calling"):
            filters["supports_function_calling"] = True

        min_quality = requirements.get("min_quality", 6.0)
        if current_model in self.pricing:
            current_quality = self.pricing[current_model].quality_score
            # Only recommend models within 1 quality point
            min_quality = max(min_quality, current_quality - 1)

        filters["min_quality"] = min_quality

        comparison = self.compare(input_tokens, output_tokens, filters=filters)

        recommendations = []
        for cost in comparison.comparisons:
            if cost.model_id == current_model:
                continue

            savings = current_cost.total_cost - cost.total_cost
            if savings <= 0:
                continue

            savings_percent = (savings / current_cost.total_cost) * 100

            # Determine quality impact
            quality_diff = self.pricing[current_model].quality_score - self.pricing[cost.model_id].quality_score
            if abs(quality_diff) <= 0.5:
                quality_impact = "minimal"
            elif abs(quality_diff) <= 1.5:
                quality_impact = "moderate"
            else:
                quality_impact = "significant"

            recommendations.append(OptimizationRecommendation(
                current_model=current_model,
                recommended_model=cost.model_id,
                reason=f"Cost ${cost.total_cost:.6f} vs ${current_cost.total_cost:.6f}",
                estimated_savings_percent=round(savings_percent, 1),
                estimated_savings_usd=round(savings, 6),
                quality_impact=quality_impact,
                confidence=0.9
            ))

        return sorted(recommendations, key=lambda r: r.estimated_savings_percent, reverse=True)

    def get_model_info(self, model: str) -> Optional[ModelPricing]:
        """Get pricing information for a model"""
        return self.pricing.get(model)

    def get_all_models(self) -> List[str]:
        """Get all available models"""
        return list(self.pricing.keys())

    def get_models_by_provider(self, provider: Provider) -> List[str]:
        """Get all models from a provider"""
        return [m for m, p in self.pricing.items() if p.provider == provider]

    def get_cheapest(self, supports_vision: bool = False) -> str:
        """Get the cheapest model"""
        models = list(self.pricing.keys())
        if supports_vision:
            models = [m for m in models if self.pricing[m].supports_vision]

        return min(models, key=lambda m: (
            self.pricing[m].input_price_per_million +
            self.pricing[m].output_price_per_million
        ) / 2)

    def _get_pricing(self, model: str) -> ModelPricing:
        """Get pricing for a model, raise if not found"""
        if model not in self.pricing:
            raise ValueError(f"Unknown model: {model}")
        return self.pricing[model]


class PricingDatabase:
    """Interface to the pricing database"""

    @staticmethod
    def get_all_models() -> Dict[str, ModelPricing]:
        """Get all models with pricing"""
        return CostCalculator.PRICING_DB.copy()

    @staticmethod
    def get_models_by_provider(provider: Provider) -> Dict[str, ModelPricing]:
        """Get models by provider"""
        return {
            k: v for k, v in CostCalculator.PRICING_DB.items()
            if v.provider == provider
        }

    @staticmethod
    def get_cheapest_model() -> str:
        """Get the absolute cheapest model"""
        return min(
            CostCalculator.PRICING_DB.keys(),
            key=lambda m: (
                CostCalculator.PRICING_DB[m].input_price_per_million +
                CostCalculator.PRICING_DB[m].output_price_per_million
            ) / 2
        )

    @staticmethod
    def get_fastest_model(min_quality: float = 6.0) -> str:
        """Get the fastest model above quality threshold"""
        candidates = [
            m for m, p in CostCalculator.PRICING_DB.items()
            if p.quality_score >= min_quality
        ]
        return min(candidates, key=lambda m: CostCalculator.PRICING_DB[m].estimated_latency_ms)


# Convenience functions
def get_all_models() -> Dict[str, ModelPricing]:
    return PricingDatabase.get_all_models()


def get_models_by_provider(provider: Provider) -> Dict[str, ModelPricing]:
    return PricingDatabase.get_models_by_provider(provider)


def get_cheapest_model() -> str:
    return PricingDatabase.get_cheapest_model()


def get_fastest_model(min_quality: float = 6.0) -> str:
    return PricingDatabase.get_fastest_model(min_quality)
