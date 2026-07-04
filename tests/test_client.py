import logging

from client import AnthropicClient, cost


def test_anthropic_tool_schema_maps_parameters_to_input_schema():
    client = AnthropicClient(model="m", api_key="dummy", rates={})
    canonical = [
        {
            "name": "t",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert client._to_anthropic_tools(canonical) == [
        {
            "name": "t",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_cost_uses_price_row_when_present():
    rates = {"m": {"input": 1.0, "output": 2.0}}
    # 1M input @ $1 + 1M output @ $2 = $3, priced per 1M tokens.
    assert cost(rates, "m", 1_000_000, 1_000_000) == 3.0


def test_cost_missing_model_meters_zero_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = cost({}, "unknown", 1_000_000, 1_000_000)
    assert result == 0.0
    assert any("no price" in r.message for r in caplog.records)
