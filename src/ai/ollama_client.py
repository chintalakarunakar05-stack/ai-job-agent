"""Ollama API Client wrapper.

Provides provider abstraction for local LLM inference via Ollama,
supporting prompt-instructed JSON extraction, configurable models, and robust error handling.
"""

import logging
from typing import Any, Dict, Optional
import ollama

from src.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama LLM provider."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama server (e.g. http://localhost:11434).
            model: Model identifier (e.g. qwen3.5:0.8b).
            timeout: Maximum response timeout in seconds.
                     Defaults to settings.OLLAMA_TIMEOUT_SECONDS (configurable via .env).
        """
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else settings.OLLAMA_TIMEOUT_SECONDS
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.1,
    ) -> str:
        """Submit generation request to Ollama.

        Args:
            prompt: User prompt text.
            system: Optional system instruction.
            json_mode: Kept for interface compatibility; JSON instruction is embedded in prompt.
            temperature: Sampling temperature.

        Returns:
            Raw response text from model.

        Raises:
            ConnectionError: If Ollama server cannot be reached.
            TimeoutError: If request exceeds timeout.
            RuntimeError: If Ollama returns failure status.
        """
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": 250,  # Cap output tokens to ensure fast completion
        }

        try:
            logger.debug(f"Sending prompt to Ollama ({self.model})")
            # Combine system instruction into prompt for reliable compact models
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                stream=False,
                think=False,
                options=options,
            )
            return response.get("response", "").strip()
        except ollama.ResponseError as exc:
            logger.error(f"Ollama server response error: {exc}")
            raise RuntimeError(f"Ollama error: {exc}") from exc
        except Exception as exc:
            str_exc = str(exc).lower()
            if "timeout" in str_exc or "timed out" in str_exc:
                logger.error(f"Ollama request timed out after {self.timeout}s: {exc}")
                raise TimeoutError(f"Ollama inference timed out after {self.timeout}s") from exc
            if "connection" in str_exc or "connect" in str_exc:
                logger.error(f"Cannot connect to Ollama server at {self.base_url}: {exc}")
                raise ConnectionError(f"Could not connect to Ollama at {self.base_url}") from exc
            logger.error(f"Ollama call failed: {exc}")
            raise RuntimeError(f"Ollama inference error: {exc}") from exc

    def check_health(self) -> bool:
        """Check if Ollama server is reachable and responsive."""
        try:
            self.client.list()
            return True
        except Exception:
            return False
