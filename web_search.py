"""
retrieval/web_search.py
────────────────────────────────────────────────────────────────────────────────
Tavily web search — used as fallback when Qdrant retrieval is graded irrelevant.

KEY CONCEPT: Why Tavily over Google/Bing API for RAG
─────────────────────────────────────────────────────
General search APIs (Google, Bing) return:
  - Full HTML pages → you need to parse and clean them
  - SEO-optimized results → may not be factually dense
  - Pagination links, nav menus, ads → noise in your context

Tavily is purpose-built for RAG pipelines:
  - Returns pre-extracted clean text from each result
  - Ranks by information density, not SEO
  - Has a `search_depth="advanced"` mode that scrapes the full page
  - Returns `max_results` controlled cleanly
  - Has a free tier: 1000 searches/month

For legal queries specifically, Tavily often returns:
  - Law school notes, bar council summaries
  - Government India Code pages
  - Legal news sites (Livelaw, Barandbench)

KNOWLEDGE REFINEMENT:
After web results come back, we run them through a Gemini call
(KNOWLEDGE_REFINE_PROMPT) to strip noise and extract only legally
relevant facts. This is the "refinement" step in CRAG's full pipeline.
"""

import os
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@dataclass
class WebResult:
    """A single web search result with cleaned content."""
    url: str
    title: str
    content: str            # clean text extracted by Tavily
    score: float = 0.0      # Tavily relevance score

    def __repr__(self):
        return f"WebResult(score={self.score:.3f}, title='{self.title[:50]}', url={self.url})"


class TavilyWebSearch:
    """
    Tavily search client with legal-domain query optimization.

    Usage:
        searcher = TavilyWebSearch()
        results = searcher.search("IPC Section 302 punishment for murder India")
        refined = searcher.refine_results(results)   # extract legal facts only
    """

    def __init__(self, api_key: str | None = None, max_results: int = 3):
        from tavily import TavilyClient
        key = api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            raise ValueError("TAVILY_API_KEY not set. Check your .env file.")
        self.client = TavilyClient(api_key=key)
        self.max_results = max_results
        logger.info(f"TavilyWebSearch ready (max_results={max_results})")

    def _build_legal_query(self, question: str) -> str:
        """
        Append Indian legal context terms to improve search result quality.

        Raw: "What is the punishment for culpable homicide?"
        Improved: "What is the punishment for culpable homicide? India IPC law"

        This biases Tavily toward Indian legal sources rather than
        US/UK law results which dominate English-language legal search.
        """
        legal_suffix = "India law legal provision"
        if "ipc" in question.lower() or "penal code" in question.lower():
            legal_suffix = "Indian Penal Code IPC"
        elif "rti" in question.lower() or "information" in question.lower():
            legal_suffix = "Right to Information Act India RTI"
        elif "constitution" in question.lower() or "article" in question.lower():
            legal_suffix = "Constitution of India"
        elif "crpc" in question.lower() or "procedure" in question.lower():
            legal_suffix = "CrPC Code of Criminal Procedure India"

        return f"{question} {legal_suffix}"

    def search(self, question: str) -> list[WebResult]:
        """
        Search the web for information relevant to a legal question.
        Uses Tavily's advanced depth for more complete page extraction.
        """
        query = self._build_legal_query(question)
        logger.info(f"Web search: '{query[:80]}...'")

        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",   # full page content, not just snippet
                max_results=self.max_results,
                include_raw_content=False,  # use Tavily's cleaned version
            )
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

        results = []
        for r in response.get("results", []):
            results.append(WebResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            ))

        logger.info(f"Web search returned {len(results)} results")
        for r in results:
            logger.debug(f"  {r}")

        return results

    def refine_results(self, results: list[WebResult], question: str) -> str:
        """
        Extract only legally relevant facts from raw web results.
        Uses Gemini with the KNOWLEDGE_REFINE_PROMPT to strip noise.

        Returns a clean bullet-point string ready for the RAG prompt context.
        """
        from llm import GeminiLLM
        from prompts import KNOWLEDGE_REFINE_PROMPT

        if not results:
            return "No web results found."

        # Combine all web content into one block
        combined = "\n\n---\n\n".join([
            f"SOURCE: {r.title} ({r.url})\n{r.content}"
            for r in results
        ])

        llm = GeminiLLM()
        prompt = KNOWLEDGE_REFINE_PROMPT.format(web_content=combined[:4000])
        refined = llm.generate(prompt, temperature=0.0, max_tokens=1024)

        logger.info("Web content refined into legal facts")
        return refined

    def search_and_refine(self, question: str) -> tuple[list[WebResult], str]:
        """
        Convenience method: search + refine in one call.
        Returns (raw_results, refined_text).
        CRAG uses the refined_text as context for the final generation.
        """
        results = self.search(question)
        refined = self.refine_results(results, question)
        return results, refined
