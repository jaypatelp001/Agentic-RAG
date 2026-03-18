# Agentic RAG Learning Project

This repository serves as a learning project exploring advanced **Retrieval-Augmented Generation (RAG)** patterns using **LangGraph**, **Gemini**, and **Qdrant**.

## Overview
The goal of this project is to progress from a simple baseline RAG implementation towards autonomous agentic patterns that self-correct and gracefully handle bad retrievals.

### Implemented Patterns
1. **Naive RAG (Phase 2):**  
   The foundational baseline (`navie_rag.py`). Represents standard "embed query → retrieve top k → generate". Subject to classic RAG failure modes (hallucinations, multi-hop misses).
2. **Corrective RAG (CRAG) (Phase 3):**  
   Uses `LangGraph` to grade retrieved chunks using an LLM evaluator (`crag_grader.py`). If chunks are deemed irrelevant, the graph falls back to a web search via **Tavily** (`web_search.py`) to gather facts from outside the vector database.
3. **Self-RAG (Phase 4):**  
   Introduces an active reflection loop within the graph (`self_rag.py`). Uses LLM reflection tokens (`IsRETRIEVE`, `IsREL`, `IsSUP`, `IsUSE`) to independently control generation routes. If the drafted answer hallucinates unsupported claims or fails to answer the question, the system loops back and regenerates using stricter prompts.

## Tech Stack
- **LLM:** Gemini 2.5 Flash / Gemini text-embedding models (`google-generativeai`)
- **Orchestration:** LangGraph (StateGraphs and conditional routing)
- **Vector DB:** Qdrant Database (Local)
- **Web Fallback:** Tavily API

## Getting Started
Ensure you have the appropriate environment variables set in an `.env` file:
- `GEMINI_API_KEY`
- `TAVILY_API_KEY`

Run any of the pattern demos locally:
```bash
python .\crag.py
python .\self_rag.py
```
