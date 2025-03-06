import os
import time
import json
import anthropic
import openai
from logging import getLogger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = getLogger(__name__)


class LLMService:
    def __init__(self, config):
        self.config = config
        self.provider = config.LLM_PROVIDER.lower()
        self.setup_client()

    def setup_client(self):
        """Set up the appropriate LLM client based on configuration"""
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
            self.model = self.config.ANTHROPIC_MODEL
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
            self.model = self.config.OPENAI_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def query(self, prompt, temperature=0.1):
        """Send a query to the LLM and get a response"""
        try:
            if self.provider == "anthropic":
                return self._query_anthropic(prompt, temperature)
            elif self.provider == "openai":
                return self._query_openai(prompt, temperature)
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            raise

    def _query_anthropic(self, prompt, temperature):
        """Query the Anthropic Claude API"""
        message = self.client.messages.create(
            model=self.model,
            temperature=temperature,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _query_openai(self, prompt, temperature):
        """Query the OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are an expert CV analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
