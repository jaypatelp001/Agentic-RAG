"""
agent/llm.py
────────────────────────────────────────────────────────────────────────────────
Gemini 1.5 Pro wrapper for all generation tasks in LexRAG.

KEY CONCEPT: Why we wrap the LLM instead of calling it directly
────────────────────────────────────────────────────────────────
Every pattern (CRAG, Self-RAG, RAG-Fusion) needs the LLM for different tasks:
  - Final answer generation    (long, detailed)
  - Retrieval grading          (short: "yes" / "no")
  - Query decomposition        (structured JSON output)
  - Reflection token generation (Self-RAG tokens: IsREL, IsSUP, IsUSE)

Wrapping lets us configure temperature, max_tokens, and output format
per task — rather than using one config for everything.

TEMPERATURE GUIDE for RAG:
  0.0 → grading, routing, structured output (deterministic)
  0.2 → final legal answer generation (factual, low creativity)
  0.7 → query decomposition / brainstorming (more varied rewrites)
"""

import os
from typing import Optional
from loguru import logger

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"


class GeminiLLM:
    """
    Thin wrapper around Gemini 2.5 Flash for different RAG tasks.

    Usage:
        llm = GeminiLLM()
        answer = llm.generate(prompt, temperature=0.2)
        grade  = llm.grade(prompt)          # temp=0.0, short output
        decomp = llm.decompose(prompt)      # temp=0.7, varied output
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"GeminiLLM ready — model: {GEMINI_MODEL}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Main generation — used for final answer synthesis.
        temperature=0.2 keeps legal answers factual and grounded.
        """
        config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = self.model.generate_content(prompt, generation_config=config)
        try:
            return response.text.strip()
        except ValueError:
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            reason = response.candidates[0].finish_reason if response.candidates else "unknown"
            logger.error(f"Generate block error: finish_reason={reason}")
            return f"Error: Generation stopped with reason {reason}"

    def grade(self, prompt: str) -> str:
        """
        Binary grading — used in CRAG to score retrieved chunks.
        temperature=0.0 forces deterministic yes/no output.
        max_tokens=10 — we only need a single word.
        """
        config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=10,
        )
        response = self.model.generate_content(prompt, generation_config=config)
        try:
            return response.text.strip().lower()
        except ValueError:
            return "no"  # Fail safe to discard on blocked content

    def decompose(self, prompt: str) -> str:
        """
        Query decomposition / rewriting — used in RAG-Fusion.
        temperature=0.7 produces varied query rewrites.
        """
        config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=512,
        )
        response = self.model.generate_content(prompt, generation_config=config)
        try:
            return response.text.strip()
        except ValueError:
            return ""