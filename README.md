# AI Engineering — From Foundations to Production

A comprehensive, expert-depth AI engineering curriculum. 22 sections, ~530 notebooks, 21 capstone projects, 20 landmark paper implementations.

Every topic follows: **foundations → advanced → expert → build**

## Sections

| # | Section | Topics | Focus |
|---|---------|--------|-------|
| 01 | LLMs Deep Internals | 8 | Tokenization, attention, transformers, pretraining, alignment, inference, reasoning |
| 02 | Hugging Face Ecosystem | 6 | Transformers, Datasets, PEFT, TRL, Accelerate, Spaces |
| 03 | Foundation Models | 8 | Model families, selection, context engineering, prompting, local inference |
| 04 | RAG | 12 | Embeddings, chunking, vector stores, retrieval, advanced patterns, GraphRAG, eval |
| 05 | LangChain Ecosystem | 15 | LangChain core, LangGraph deep dive, LangSmith, observability |
| 06 | Agent Frameworks | 11 | LangGraph/OpenAI/Claude (expert), CrewAI/AutoGen/DSPy/SK/Haystack (working), MCP |
| 07 | Agent Architectures | 9 | ReAct, planning, multi-agent, memory, coding agents, research agents, text-to-SQL |
| 08 | Vision | 6 | CNNs, ViTs, detection, segmentation, generation, 3D |
| 09 | Audio | 5 | Fundamentals, ASR, speakers, TTS, music generation |
| 10 | Video | 4 | Temporal modeling, understanding, tracking, generation |
| 11 | Multimodal | 4 | CLIP, VLMs, document AI, any-to-any |
| 12 | Fine-Tuning | 9 | Data engineering, PEFT, full FT, eval, synthetic data, distillation |
| 13 | Evaluation | 6 | LLM eval, RAG eval, agent eval, safety eval, production eval |
| 14 | Safety & Alignment | 5 | Prompt security, guardrails, alignment, responsible AI, compliance |
| 15 | AI System Design | 9 | Design patterns, cost, caching, latency, resilience, cloud, UX, case studies |
| 16 | Data Engineering for AI | 3 | Document parsing, pipelines, web data |
| 17 | Testing & CI for AI | 4 | Unit testing, integration testing, CI/CD, load testing |
| 18 | Workflow Orchestration | 3 | Temporal, event-driven, pipeline orchestration |
| 19 | Real-Time AI | 4 | Streaming, voice pipelines, real-time vision, live multimodal |
| 20 | Edge & On-Device | 3 | Compression, on-device inference, hybrid architectures |
| 21 | Deployment & MLOps | 5 | Serving, optimization, infrastructure, MLOps, monitoring |
| 22 | Research Papers | 20 | Landmark paper implementations from scratch |

## How to Use

Each topic directory contains:
- `foundations.ipynb` — core concepts, implement from scratch where possible
- `advanced.ipynb` — production-relevant depth, real-world patterns
- `expert.ipynb` — deep internals, the stuff that makes you the go-to person
- `build.ipynb` — hands-on project applying everything in the topic

Start with Section 01 and work through sequentially, or jump to any section that matches your current needs.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Design Spec

Full curriculum design: [docs/superpowers/specs/2026-04-11-ai-engineering-curriculum-design.md](docs/superpowers/specs/2026-04-11-ai-engineering-curriculum-design.md)
