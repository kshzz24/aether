from client import LLMClient, OpenAICompatibleClient
import logging
from openai import APIConnectionError


class GatewayClient(LLMClient):
    def __init__(self, gateway_url, model, fallback: LLMClient, rates, api_key=""):

        self._fallback = fallback
        # The gateway needs no auth (it holds the provider keys), but the OpenAI
        # SDK refuses to construct with an empty api_key. Pass a placeholder.
        self._upstream = OpenAICompatibleClient(
            model=model, api_key=api_key or "no-auth", base_url=gateway_url, rates=rates
        )

    async def create(self, messages, tools, system):
        try:
            return await self._upstream.create(
                messages=messages, tools=tools, system=system
            )
        except APIConnectionError:
            logging.warning("Gateway unreachable, degrading to passthrough")
            return await self._fallback.create(
                messages=messages, tools=tools, system=system
            )
