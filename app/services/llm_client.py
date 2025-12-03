"""
🤖 LLM CLIENT SERVICE - Groq API Integration

Service générique pour appeler des LLM via Groq (llama-3.3-70b-versatile, etc.)

Author: AI Assistant
Date: December 2025
Version: 2.0 - Groq
"""

import logging
import httpx
from typing import List, Dict, Optional
from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client générique pour appeler des LLM via l'API Groq"""

    def __init__(self):
        self.settings = get_settings()
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = getattr(self.settings, "groq_api_key", "")
        self.model = getattr(self.settings, "groq_model", "llama-3.3-70b-versatile")
        
        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not configured - LLM features will not work")
        else:
            logger.info(f"✅ LLM Client initialized with Groq model: {self.model}")

    def is_available(self) -> bool:
        """Vérifie si le service LLM est disponible"""
        return bool(self.api_key and self.model)

    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 400,
        stream: bool = False
    ) -> str:
        """
        Appelle le LLM via OpenRouter API
        
        Args:
            messages: Liste de messages au format ChatGPT
                      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Température (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre maximum de tokens à générer
            stream: Streaming activé (non implémenté pour le moment)
            
        Returns:
            Le texte de la réponse du LLM
            
        Raises:
            ValueError: Si l'API key n'est pas configurée
            httpx.HTTPError: Si la requête échoue
        """
        if not self.is_available():
            raise ValueError(
                "LLM service not available. Please configure GROQ_API_KEY "
                "and GROQ_MODEL in your .env file"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(f"🤖 Calling Groq API: model={self.model}, temp={temperature}, max_tokens={max_tokens}")
        logger.debug(f"📝 Messages: {len(messages)} messages, total chars: {sum(len(m['content']) for m in messages)}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )

                # Log de debug pour statut HTTP
                logger.debug(f"📡 HTTP Status: {response.status_code}")

                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(
                        f"❌ Groq API error: {response.status_code} - {error_detail}"
                    )
                    raise httpx.HTTPStatusError(
                        f"Groq API returned status {response.status_code}: {error_detail}",
                        request=response.request,
                        response=response
                    )

                response_data = response.json()
                
                # Extraction du contenu de la réponse
                if "choices" not in response_data or len(response_data["choices"]) == 0:
                    logger.error(f"❌ Invalid response structure: {response_data}")
                    raise ValueError("Invalid response from Groq API: no choices")

                content = response_data["choices"][0]["message"]["content"]
                
                # Log usage si disponible
                if "usage" in response_data:
                    usage = response_data["usage"]
                    logger.info(
                        f"💰 Token usage: prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}, "
                        f"total={usage.get('total_tokens', 0)}"
                    )

                logger.info(f"✅ LLM response received: {len(content)} chars")
                return content

        except httpx.TimeoutException:
            logger.error("⏱️ Timeout while calling Groq API")
            raise
        except httpx.HTTPError as e:
            logger.error(f"🌐 HTTP error while calling Groq API: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error calling LLM: {e}")
            raise


# Instance globale (singleton pattern)
llm_client = LLMClient()


async def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.4,
    max_tokens: int = 400
) -> str:
    """
    Fonction helper pour appeler le LLM via l'instance globale
    
    Usage:
        from app.services.llm_client import call_llm
        
        response = await call_llm(
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"}
            ],
            temperature=0.7,
            max_tokens=150
        )
    """
    return await llm_client.call_llm(messages, temperature, max_tokens)


def is_llm_available() -> bool:
    """Vérifie si le service LLM est disponible"""
    return llm_client.is_available()
