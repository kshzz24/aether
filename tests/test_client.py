from client import AnthropicClient


def test_anthropic_tool_schema_maps_parameters_to_input_schema():
    client = AnthropicClient(model="m", api_key="dummy")
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
