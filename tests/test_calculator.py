"""
Tests for LLM Cost Calculator
"""

import pytest
from llm_cost_calculator import (
    CostCalculator,
    ModelPricing,
    RequestCost,
    CostComparison,
    OptimizationRecommendation,
    BudgetTracker,
    BudgetAlert,
    CostForecast,
    Provider,
    get_all_models,
    get_models_by_provider,
    get_cheapest_model,
    get_fastest_model,
)


class TestCostCalculator:
    """Test CostCalculator class"""

    @pytest.fixture
    def calculator(self):
        return CostCalculator()

    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert len(calculator.pricing) > 0
        assert "gpt-4o" in calculator.pricing
        assert isinstance(calculator.budget_tracker, BudgetTracker)
        assert isinstance(calculator.forecast, CostForecast)

    def test_calculate_gpt4o(self, calculator):
        """Test calculating cost for GPT-4o"""
        cost = calculator.calculate("gpt-4o", 1000, 500)
        assert cost.model_id == "gpt-4o"
        assert cost.provider == Provider.OPENAI
        assert cost.input_tokens == 1000
        assert cost.output_tokens == 500
        assert cost.total_cost > 0

    def test_calculate_pricing_accuracy(self, calculator):
        """Test pricing calculation accuracy"""
        # GPT-4o: $2.50/1M input, $10.00/1M output
        cost = calculator.calculate("gpt-4o", 1_000_000, 1_000_000)
        assert cost.input_cost == 2.50
        assert cost.output_cost == 10.00
        assert cost.total_cost == 12.50

    def test_calculate_claude(self, calculator):
        """Test calculating cost for Claude"""
        cost = calculator.calculate("claude-3-opus", 1000, 500)
        assert cost.model_id == "claude-3-opus"
        assert cost.provider == Provider.ANTHROPIC

    def test_calculate_unknown_model(self, calculator):
        """Test calculating cost for unknown model"""
        with pytest.raises(ValueError):
            calculator.calculate("unknown-model-xyz", 1000, 500)

    def test_calculate_with_record(self, calculator):
        """Test calculating with recording enabled"""
        cost = calculator.calculate("gpt-4o", 1000, 500, record=True)
        assert len(calculator.forecast.history) == 1

    def test_compare_all_models(self, calculator):
        """Test comparing costs across all models"""
        comparison = calculator.compare(1000, 500)
        assert len(comparison.comparisons) > 0
        assert comparison.cheapest.total_cost < comparison.most_expensive.total_cost

    def test_compare_filtered_models(self, calculator):
        """Test comparing with filters"""
        comparison = calculator.compare(
            1000, 500,
            filters={"supports_vision": True}
        )
        # All comparisons should support vision
        for cost in comparison.comparisons:
            pricing = calculator.get_model_info(cost.model_id)
            assert pricing.supports_vision is True

    def test_optimize_recommendations(self, calculator):
        """Test getting optimization recommendations"""
        recommendations = calculator.optimize(
            current_model="gpt-4",
            input_tokens=1000,
            output_tokens=500
        )
        assert isinstance(recommendations, list)

    def test_get_model_info(self, calculator):
        """Test getting model info"""
        info = calculator.get_model_info("gpt-4o")
        assert isinstance(info, ModelPricing)
        assert info.model_id == "gpt-4o"
        assert info.provider == Provider.OPENAI

    def test_get_model_info_unknown(self, calculator):
        """Test getting info for unknown model"""
        info = calculator.get_model_info("unknown")
        assert info is None

    def test_get_all_models(self, calculator):
        """Test getting all models"""
        models = calculator.get_all_models()
        assert len(models) > 0
        assert "gpt-4o" in models

    def test_get_models_by_provider(self, calculator):
        """Test getting models by provider"""
        openai_models = calculator.get_models_by_provider(Provider.OPENAI)
        assert "gpt-4o" in openai_models
        assert "gpt-3.5-turbo" in openai_models

    def test_get_cheapest(self, calculator):
        """Test getting cheapest model"""
        cheapest = calculator.get_cheapest()
        assert cheapest in calculator.pricing

    def test_get_cheapest_with_vision(self, calculator):
        """Test getting cheapest model with vision support"""
        cheapest = calculator.get_cheapest(supports_vision=True)
        pricing = calculator.get_model_info(cheapest)
        assert pricing.supports_vision is True


class TestModelPricing:
    """Test ModelPricing dataclass"""

    def test_model_pricing_properties(self):
        """Test ModelPricing calculated properties"""
        pricing = ModelPricing(
            model_id="test-model",
            provider=Provider.OPENAI,
            input_price_per_million=10.0,
            output_price_per_million=20.0,
            context_window=128000,
        )
        assert pricing.input_price_per_token == 10.0 / 1_000_000
        assert pricing.output_price_per_token == 20.0 / 1_000_000

    def test_model_pricing_with_all_fields(self):
        """Test ModelPricing with all fields"""
        pricing = ModelPricing(
            model_id="test",
            provider=Provider.ANTHROPIC,
            input_price_per_million=1.0,
            output_price_per_million=2.0,
            context_window=200000,
            max_output_tokens=8192,
            supports_streaming=True,
            supports_function_calling=True,
            supports_vision=True,
            estimated_latency_ms=300,
            quality_score=9.0
        )
        assert pricing.supports_streaming is True
        assert pricing.quality_score == 9.0


class TestRequestCost:
    """Test RequestCost dataclass"""

    def test_request_cost_properties(self):
        """Test RequestCost calculated properties"""
        cost = RequestCost(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03
        )
        assert cost.total_tokens == 1500
        assert cost.cost_per_1k_tokens == (0.03 / 1500) * 1000


class TestBudgetTracker:
    """Test BudgetTracker"""

    @pytest.fixture
    def tracker(self):
        return BudgetTracker()

    def test_set_budget(self, tracker):
        """Test setting a budget"""
        tracker.set_budget("monthly", 100.00)
        assert tracker.budgets["monthly"] == 100.00

    def test_record_cost(self, tracker):
        """Test recording a cost"""
        tracker.set_budget("test", 100.00)
        alert = tracker.record_cost("test", 10.00)
        assert tracker.spent["test"] == 10.00

    def test_record_triggers_alert(self, tracker):
        """Test that recording triggers alerts at thresholds"""
        tracker.set_budget("test", 100.00)

        # Should trigger at 50%
        tracker.record_cost("test", 50.00)
        alert = tracker._check_alerts("test")
        assert alert is not None
        assert alert.threshold_value == 50

    def test_get_remaining(self, tracker):
        """Test getting remaining budget"""
        tracker.set_budget("test", 100.00)
        tracker.record_cost("test", 25.00)
        assert tracker.get_remaining("test") == 75.00

    def test_get_percent_used(self, tracker):
        """Test getting percent used"""
        tracker.set_budget("test", 100.00)
        tracker.record_cost("test", 25.00)
        assert tracker.get_percent_used("test") == 25.0

    def test_get_status(self, tracker):
        """Test getting budget status"""
        tracker.set_budget("test", 100.00)
        tracker.record_cost("test", 25.00)
        status = tracker.get_status("test")
        assert status["budget"] == 100.00
        assert status["spent"] == 25.00
        assert status["remaining"] == 75.00
        assert status["percent_used"] == 25.0


class TestBudgetAlert:
    """Test BudgetAlert dataclass"""

    def test_budget_alert_creation(self):
        """Test creating BudgetAlert"""
        alert = BudgetAlert(
            budget_name="monthly",
            threshold_type="percent",
            threshold_value=75,
            current_spent=75.00,
            current_percent=75.0,
            message="Budget is 75% used"
        )
        assert alert.budget_name == "monthly"
        assert alert.threshold_value == 75


class TestCostForecast:
    """Test CostForecast"""

    @pytest.fixture
    def forecast(self):
        return CostForecast()

    def test_add_request(self, forecast):
        """Test adding a request to history"""
        cost = RequestCost(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03
        )
        forecast.add_request(cost)
        assert len(forecast.history) == 1

    def test_forecast_daily(self, forecast):
        """Test daily cost forecasting"""
        cost = RequestCost(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.03,
            output_cost=0.01,
            total_cost=0.04
        )
        forecast.add_request(cost)

        daily = forecast.forecast_daily("gpt-4o", 100)
        assert daily == 0.04 * 100

    def test_forecast_monthly(self, forecast):
        """Test monthly cost forecasting"""
        cost = RequestCost(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.03,
            output_cost=0.01,
            total_cost=0.04
        )
        forecast.add_request(cost)

        monthly = forecast.forecast_monthly("gpt-4o", 100)
        assert monthly == 0.04 * 100 * 30

    def test_forecast_tokens_per_month(self, forecast):
        """Test token usage forecasting"""
        cost = RequestCost(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.03,
            output_cost=0.01,
            total_cost=0.04
        )
        forecast.add_request(cost)

        tokens = forecast.forecast_tokens_per_month("gpt-4o", 100)
        assert tokens["input_tokens"] == 1000 * 100 * 30
        assert tokens["output_tokens"] == 500 * 100 * 30
        assert tokens["total_tokens"] == 1500 * 100 * 30


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation dataclass"""

    def test_recommendation_creation(self):
        """Test creating optimization recommendation"""
        rec = OptimizationRecommendation(
            current_model="gpt-4",
            recommended_model="gpt-4o",
            reason="Lower cost",
            estimated_savings_percent=50.0,
            estimated_savings_usd=0.01,
            quality_impact="minimal",
            confidence=0.9
        )
        assert rec.current_model == "gpt-4"
        assert rec.recommended_model == "gpt-4o"
        assert rec.estimated_savings_percent == 50.0


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_all_models(self):
        """Test getting all models"""
        models = get_all_models()
        assert isinstance(models, dict)
        assert "gpt-4o" in models

    def test_get_models_by_provider(self):
        """Test getting models by provider"""
        models = get_models_by_provider(Provider.OPENAI)
        assert isinstance(models, dict)
        # All should be OpenAI
        for pricing in models.values():
            assert pricing.provider == Provider.OPENAI

    def test_get_cheapest_model(self):
        """Test getting cheapest model"""
        cheapest = get_cheapest_model()
        assert isinstance(cheapest, str)

    def test_get_fastest_model(self):
        """Test getting fastest model"""
        fastest = get_fastest_model(min_quality=7.0)
        assert isinstance(fastest, str)


class TestCostComparison:
    """Test CostComparison dataclass"""

    def test_comparison_properties(self):
        """Test CostComparison has required fields"""
        from llm_cost_calculator import CostComparison, RequestCost, Provider

        costs = [
            RequestCost("gpt-4o", Provider.OPENAI, 1000, 500, 0.01, 0.02, 0.03),
            RequestCost("claude-3-haiku", Provider.ANTHROPIC, 1000, 500, 0.005, 0.01, 0.015)
        ]

        comparison = CostComparison(
            request_description="test",
            input_tokens=1000,
            output_tokens=500,
            comparisons=costs,
            cheapest=costs[1],
            most_expensive=costs[0],
            savings_vs_cheapest=0.015
        )
        assert comparison.input_tokens == 1000
        assert comparison.savings_vs_cheapest == 0.015


class TestProviderEnum:
    """Test Provider enum values"""

    def test_provider_values(self):
        """Test Provider enum has correct values"""
        assert Provider.OPENAI == "openai"
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.GOOGLE == "google"
        assert Provider.META == "meta"
        assert Provider.MISTRAL == "mistral"
        assert Provider.DEEPSEEK == "deepseek"
        assert Provider.COHERE == "cohere"
