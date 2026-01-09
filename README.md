# LLM Cost Calculator

Comprehensive cost calculation and optimization for LLM APIs.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, Cohere
- **Cost Comparison**: Compare costs across 20+ models instantly
- **Optimization Recommendations**: Get actionable suggestions to reduce costs
- **Budget Tracking**: Set budgets and receive alerts when approaching limits
- **Cost Forecasting**: Project future costs based on usage patterns
- **Model Database**: 20+ models with up-to-date pricing

## Installation

```bash
pip install llm-cost-calculator
```

## Quick Start

```python
from llm_cost_calculator import CostCalculator, estimate_cost

# Create calculator
calc = CostCalculator()

# Calculate single request cost
cost = calc.calculate(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500
)
print(f"Cost: ${cost.total_cost:.6f}")
print(f"Input: ${cost.input_cost:.6f}, Output: ${cost.output_cost:.6f}")

# Compare across all models
comparison = calc.compare(input_tokens=1000, output_tokens=500)
print(f"\nCheapest: {comparison.cheapest.model_id} @ ${comparison.cheapest.total_cost:.6f}")
print(f"Most expensive: {comparison.most_expensive.model_id} @ ${comparison.most_expensive.total_cost:.6f}")
print(f"Savings: ${comparison.savings_vs_cheapest:.6f}")

# Get optimization recommendations
recommendations = calc.optimize(
    current_model="gpt-4",
    input_tokens=1000,
    output_tokens=500
)
for rec in recommendations:
    print(f"\n{rec.recommended_model}: Save {rec.estimated_savings_percent}%")
    print(f"  Reason: {rec.reason}")
    print(f"  Quality impact: {rec.quality_impact}")
```

## Budget Tracking

```python
from llm_cost_calculator import CostCalculator

calc = CostCalculator()

# Set monthly budget
calc.budget_tracker.set_budget("monthly", 100.00)

# Record costs
cost = calc.calculate("gpt-4o", 1000, 500)
alert = calc.budget_tracker.record_cost("monthly", cost.total_cost)

if alert:
    print(alert.message)

# Check budget status
status = calc.budget_tracker.get_status("monthly")
print(f"Spent: ${status['spent']:.2f} of ${status['budget']:.2f}")
print(f"Remaining: ${status['remaining']:.2f}")
print(f"Percent used: {status['percent_used']:.1f}%")
```

## Cost Forecasting

```python
from llm_cost_calculator import CostCalculator

calc = CostCalculator()

# Record some requests
calc.calculate("gpt-4o", 1000, 500, record=True)
calc.calculate("gpt-4o", 800, 400, record=True)

# Forecast monthly costs (assuming 100 requests/day)
daily_cost = calc.forecast.forecast_daily("gpt-4o", 100)
monthly_cost = calc.forecast.forecast_monthly("gpt-4o", 100)

print(f"Daily forecast: ${daily_cost:.2f}")
print(f"Monthly forecast: ${monthly_cost:.2f}")

# Token usage forecast
tokens = calc.forecast.forecast_tokens_per_month("gpt-4o", 100)
print(f"Monthly tokens: {tokens['total_tokens']:,}")
```

## Supported Models

### OpenAI
- gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
- gpt-3.5-turbo
- o1-preview, o1-mini

### Anthropic
- claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku
- claude-3-opus, claude-3-sonnet, claude-3-haiku

### Google
- gemini-2.0-flash-exp
- gemini-1.5-pro, gemini-1.5-flash

### Meta
- llama-3.3-70b, llama-3.1-405b, llama-3.1-70b
- llama-3-70b, llama-3-8b

### Mistral
- mistral-large
- mixtral-8x22b, mixtral-8x7b
- codestral

### DeepSeek
- deepseek-chat, deepseek-coder
- deepseek-reasoner

## API Reference

### CostCalculator

```python
CostCalculator(custom_pricing=None)
```

**Methods:**
- `calculate(model, input_tokens, output_tokens, record=False)` - Calculate request cost
- `compare(input_tokens, output_tokens, models=None, filters=None)` - Compare costs
- `optimize(current_model, input_tokens, output_tokens, requirements=None)` - Get recommendations
- `get_model_info(model)` - Get pricing for a model
- `get_all_models()` - List all models
- `get_models_by_provider(provider)` - Get models by provider
- `get_cheapest(supports_vision=False)` - Get cheapest model

### Filtering Comparisons

```python
comparison = calc.compare(
    input_tokens=1000,
    output_tokens=500,
    filters={
        "supports_vision": True,
        "supports_function_calling": True,
        "min_quality": 8.0,
        "max_latency_ms": 500
    }
)
```

### BudgetTracker

```python
budget_tracker = BudgetTracker()
budget_tracker.set_budget("monthly", 100.00)
alert = budget_tracker.record_cost("monthly", 5.00)
remaining = budget_tracker.get_remaining("monthly")
percent = budget_tracker.get_percent_used("monthly")
status = budget_tracker.get_status("monthly")
```

## License

MIT License - see LICENSE file for details.
