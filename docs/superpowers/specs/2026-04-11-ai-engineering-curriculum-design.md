# AI Engineering Curriculum — Design Spec

## Overview

A comprehensive, expert-depth AI engineering learning repository structured as Jupyter notebooks. Every topic follows a **foundations → advanced → expert → build** progression. The repo serves two purposes: personal study notes for mastering AI engineering end-to-end, and a structured teaching resource others can follow.

**Target learner:** Data scientist with ML engineering experience, aiming to become the go-to AI engineer in any organization.

**Format:** Jupyter notebooks throughout — theory and code tightly coupled. Each notebook contains explanatory markdown cells and runnable code.

**Scope:** 21 sections, ~130 topics, ~530+ notebooks, 21 capstone projects, 20 landmark paper implementations.

**Build-out strategy:** Design the full curriculum structure now, build out sections incrementally. Priority sections first: LLMs, RAG, LangChain ecosystem, agent frameworks, agent architectures.

---

## Notebook Structure Convention

Every topic directory contains:

- `foundations.ipynb` — core concepts, implement from scratch where possible
- `advanced.ipynb` — production-relevant depth, real-world patterns
- `expert.ipynb` — the stuff that makes you the person everyone comes to
- `build.ipynb` — hands-on project applying everything in the topic

Tier 2 agent framework topics use a condensed format:

- `foundations.ipynb` — core concepts and setup
- `applied.ipynb` — practical patterns and real usage
- `build.ipynb` — hands-on project

Each section ends with a **capstone/** directory containing a production-worthy project that ties the section together.

---

## Repository Structure

```
ai-engineering/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── superpowers/
│       └── specs/
│           └── this file
```

---

## Section 01: LLMs Deep Internals

The architecture, training, and inference of large language models from first principles through expert-level optimization.

```
01-llms-deep-internals/
├── 01-tokenization/
│   ├── foundations.ipynb          # BPE, WordPiece, Unigram, SentencePiece — implement each
│   ├── advanced.ipynb            # training tokenizers, multilingual, domain-specific, byte-level
│   ├── expert.ipynb              # tokenizer-model interactions, fertility, compression ratios
│   └── build.ipynb               # train + evaluate tokenizer for a specific domain
├── 02-embeddings/
│   ├── foundations.ipynb          # word2vec, GloVe — implement from scratch
│   ├── advanced.ipynb            # contextual embeddings, sentence-transformers, MTEB
│   ├── expert.ipynb              # embedding space geometry, isotropy, anisotropy, probing
│   └── build.ipynb               # fine-tune embedding model, evaluate on MTEB
├── 03-attention-mechanisms/
│   ├── foundations.ipynb          # self-attention, multi-head — implement from scratch
│   ├── advanced.ipynb            # MQA, GQA, flash attention, ring attention
│   ├── expert.ipynb              # attention sinks, attention pattern analysis, sparse attention
│   └── build.ipynb               # implement flash attention, benchmark against naive
├── 04-transformer-architectures/
│   ├── foundations.ipynb          # encoder, decoder, encoder-decoder — build each
│   ├── advanced.ipynb            # RoPE, ALiBi, RMSNorm, SwiGLU, parallel layers
│   ├── expert.ipynb              # Mixture of Experts, switch transformers, routing strategies
│   └── build.ipynb               # build a full GPT from scratch (nanoGPT extended)
├── 05-pretraining/
│   ├── foundations.ipynb          # language modeling objectives, data pipelines, tokenized datasets
│   ├── advanced.ipynb            # distributed training, DeepSpeed ZeRO, FSDP, mixed precision
│   ├── expert.ipynb              # scaling laws (Chinchilla, Kaplan), compute-optimal training, data mixtures
│   └── build.ipynb               # pretrain a small LM, verify scaling law predictions
├── 06-post-training/
│   ├── foundations.ipynb          # SFT, instruction tuning, chat templates
│   ├── advanced.ipynb            # RLHF, DPO, reward modeling, PPO
│   ├── expert.ipynb              # constitutional AI, ORPO, KTO, iterative RLHF, process reward models
│   └── build.ipynb               # full post-training pipeline: SFT → DPO → eval
├── 07-inference-engine/
│   ├── foundations.ipynb          # decoding strategies, KV caching, temperature
│   ├── advanced.ipynb            # speculative decoding, continuous batching, paged attention
│   ├── expert.ipynb              # quantization deep dive (GPTQ, AWQ, GGUF, fp8), kernel optimization
│   └── build.ipynb               # deploy with vLLM, benchmark throughput and latency
├── 08-reasoning-models/
│   ├── foundations.ipynb          # chain-of-thought, inference-time compute scaling
│   ├── advanced.ipynb            # o1/o3 patterns, reasoning traces, thinking tokens
│   ├── expert.ipynb              # process reward models, MCTS for reasoning, verification
│   └── build.ipynb               # build a reasoning pipeline with verification loops
└── capstone/
    └── build-your-own-llm/       # pretrain → SFT → DPO → quantize → serve → evaluate
```

---

## Section 02: Hugging Face Ecosystem

The open-source backbone of AI engineering. Deep mastery of the tools you use daily.

```
02-hugging-face-ecosystem/
├── 01-transformers-library/
│   ├── foundations.ipynb          # AutoModel, AutoTokenizer, pipelines, model hub
│   ├── advanced.ipynb            # custom model loading, model cards, gated models, trust_remote_code
│   ├── expert.ipynb              # extending Transformers, custom architectures, contributing models
│   └── build.ipynb               # build a model serving pipeline using Transformers
├── 02-datasets-and-data/
│   ├── foundations.ipynb          # load_dataset, streaming, dataset manipulation
│   ├── advanced.ipynb            # custom datasets, data processing pipelines, Arrow format
│   ├── expert.ipynb              # large-scale data processing, dataset deduplication, data mixing
│   └── build.ipynb               # build and publish a curated dataset to HF Hub
├── 03-peft-and-trl/
│   ├── foundations.ipynb          # PEFT library, LoRA config, TRL SFTTrainer
│   ├── advanced.ipynb            # DPO with TRL, reward modeling, multi-adapter
│   ├── expert.ipynb              # custom PEFT methods, TRL advanced configs, PPO training
│   └── build.ipynb               # full PEFT+TRL fine-tuning pipeline
├── 04-accelerate-and-training/
│   ├── foundations.ipynb          # Accelerate basics, multi-GPU, mixed precision
│   ├── advanced.ipynb            # DeepSpeed integration, FSDP, gradient accumulation
│   ├── expert.ipynb              # custom training loops, distributed strategies, profiling
│   └── build.ipynb               # train a model across multiple GPUs with Accelerate
├── 05-inference-and-deployment/
│   ├── foundations.ipynb          # HF Inference API, Inference Endpoints
│   ├── advanced.ipynb            # TGI (Text Generation Inference), TEI (Text Embeddings Inference)
│   ├── expert.ipynb              # custom inference handlers, autoscaling, cost optimization
│   └── build.ipynb               # deploy model to HF Inference Endpoints with autoscaling
├── 06-spaces-and-demos/
│   ├── foundations.ipynb          # Gradio basics, Streamlit, building demos
│   ├── advanced.ipynb            # Gradio custom components, Chainlit for chat UIs
│   ├── expert.ipynb              # Docker Spaces, persistent storage, GPU Spaces
│   └── build.ipynb               # build and deploy interactive AI demo on HF Spaces
└── capstone/
    └── hf-model-pipeline/        # train → evaluate → publish → deploy → demo on HF
```

---

## Section 03: Foundation Models

Understanding, selecting, and working with the major model families. The practical knowledge of what's available and how to use it.

```
03-foundation-models/
├── 01-model-families/
│   ├── gpt-family.ipynb          # GPT-3.5/4/4o/4.1 — architectures, strengths, API patterns
│   ├── claude-family.ipynb       # Claude 3/3.5/4 — extended thinking, tool use, system prompts
│   ├── gemini-family.ipynb       # Gemini 1.5/2.0/2.5 — long context, multimodal native
│   ├── llama-family.ipynb        # Llama 2/3/4 — open weights, fine-tuning, deployment
│   ├── mistral-family.ipynb      # Mistral/Mixtral — MoE, efficiency, Codestral
│   └── emerging-models.ipynb     # DeepSeek, Qwen, Command R, Phi, Gemma
├── 02-model-selection/
│   ├── foundations.ipynb          # benchmarks (MMLU, HumanEval, GPQA, Arena ELO), what they measure
│   ├── advanced.ipynb            # task-specific selection, cost/latency/quality tradeoffs
│   ├── expert.ipynb              # building model routers, cascading, fallback strategies
│   └── build.ipynb               # build a model router that selects optimal model per query
├── 03-context-engineering/
│   ├── foundations.ipynb          # context windows, token limits, prompt structure
│   ├── advanced.ipynb            # long-context strategies, needle-in-haystack, RAG vs stuffing
│   ├── expert.ipynb              # context caching, prompt compression, context distillation
│   └── build.ipynb               # build a context manager that optimizes token usage
├── 04-structured-output/
│   ├── foundations.ipynb          # JSON mode, function calling, response schemas
│   ├── advanced.ipynb            # constrained decoding, grammar-based generation, Outlines
│   ├── expert.ipynb              # cross-provider function calling, schema design patterns
│   └── build.ipynb               # build a universal structured output layer across providers
├── 05-prompting-mastery/
│   ├── foundations.ipynb          # zero/few-shot, CoT, system prompts, role prompting
│   ├── advanced.ipynb            # self-consistency, tree-of-thought, meta-prompting
│   ├── expert.ipynb              # DSPy optimization, prompt evolution, automatic prompt engineering
│   └── build.ipynb               # build a prompt optimization system with DSPy
├── 06-local-inference/
│   ├── foundations.ipynb          # Ollama, llama.cpp, GGUF models, setup
│   ├── advanced.ipynb            # MLX (Apple Silicon), ExLlamaV2, koboldcpp
│   ├── expert.ipynb              # performance tuning, context length, batching, GPU offloading
│   └── build.ipynb               # build a local AI stack: Ollama + Open WebUI + API
├── 07-universal-apis/
│   ├── foundations.ipynb          # LiteLLM — unified interface across providers
│   ├── advanced.ipynb            # OpenRouter, custom proxy, provider fallbacks
│   ├── expert.ipynb              # Instructor for structured output, Outlines for constrained decoding
│   └── build.ipynb               # build universal LLM gateway: LiteLLM + Instructor + caching
└── 08-confidence-and-uncertainty/
    ├── foundations.ipynb          # logprobs, calibration basics, token-level confidence
    ├── advanced.ipynb            # abstention strategies, confidence-based routing
    ├── expert.ipynb              # self-consistency for uncertainty, ensemble confidence, when the model doesn't know
    └── build.ipynb               # build confidence-aware model router (easy→small, hard→large)
```

---

## Section 04: RAG

Retrieval-Augmented Generation from fundamentals through production-grade systems. Deep enough to be the RAG expert.

```
04-rag/
├── 01-embeddings-and-indexing/
│   ├── foundations.ipynb          # embedding models, similarity metrics, vector math
│   ├── advanced.ipynb            # fine-tuning embeddings, matryoshka, binary quantization
│   ├── expert.ipynb              # ColBERT, late interaction, multi-vector representations
│   └── build.ipynb               # implement and benchmark multiple embedding strategies
├── 02-chunking/
│   ├── foundations.ipynb          # fixed, recursive, semantic chunking
│   ├── advanced.ipynb            # document-aware, agentic chunking, parent-child, propositions
│   ├── expert.ipynb              # optimal chunk size research, chunking for different doc types
│   └── build.ipynb               # build adaptive chunking pipeline with evaluation
├── 03-vector-stores/
│   ├── foundations.ipynb          # FAISS, Chroma, pgvector — internals and benchmarks
│   ├── advanced.ipynb            # Pinecone, Weaviate, Qdrant — scaling, filtering, hybrid
│   ├── expert.ipynb              # ANN algorithms (HNSW, IVF, PQ), index tuning, sharding
│   └── build.ipynb               # build a vector store from scratch with HNSW
├── 04-retrieval/
│   ├── foundations.ipynb          # dense, sparse (BM25), TF-IDF
│   ├── advanced.ipynb            # hybrid search, reciprocal rank fusion, learned sparse (SPLADE)
│   ├── expert.ipynb              # cross-encoder reranking, ColBERT reranking, listwise reranking
│   └── build.ipynb               # build a multi-stage retrieval pipeline with reranking
├── 05-query-processing/
│   ├── foundations.ipynb          # query rewriting, HyDE, step-back prompting
│   ├── advanced.ipynb            # query decomposition, multi-query, routing, classification
│   ├── expert.ipynb              # learned query expansion, query-document alignment scoring
│   └── build.ipynb               # build an intelligent query processing engine
├── 06-advanced-rag-architectures/
│   ├── foundations.ipynb          # naive RAG limitations, multi-hop reasoning
│   ├── advanced.ipynb            # self-RAG, corrective RAG (CRAG), adaptive RAG
│   ├── expert.ipynb              # speculative RAG, RAPTOR (tree-structured), modular RAG
│   └── build.ipynb               # implement CRAG and self-RAG with LangGraph
├── 07-graph-rag/
│   ├── foundations.ipynb          # knowledge graphs, entity extraction, triples, Neo4j
│   ├── advanced.ipynb            # Microsoft GraphRAG, community detection, global queries
│   ├── expert.ipynb              # hybrid graph+vector, ontology design, graph neural networks for RAG
│   └── build.ipynb               # build a full GraphRAG pipeline from scratch
├── 08-multimodal-rag/
│   ├── foundations.ipynb          # image/table extraction, multimodal embeddings
│   ├── advanced.ipynb            # ColPali, vision-based retrieval, PDF understanding
│   ├── expert.ipynb              # cross-modal retrieval, late fusion strategies
│   └── build.ipynb               # build a multimodal RAG over PDFs with tables + images
├── 09-rag-evaluation/
│   ├── foundations.ipynb          # RAGAS, faithfulness, relevancy, context precision/recall
│   ├── advanced.ipynb            # LangSmith eval, synthetic test generation, regression testing
│   ├── expert.ipynb              # custom metrics, component-wise evaluation, statistical significance
│   └── build.ipynb               # build an automated RAG eval + regression suite
├── 10-llamaindex/
│   ├── foundations.ipynb          # indexes, query engines, retrievers, node parsers
│   ├── advanced.ipynb            # custom retrievers, response synthesizers, composable indices
│   ├── expert.ipynb              # property graphs, agent-based RAG, production patterns
│   └── build.ipynb               # build same RAG system in LangChain vs LlamaIndex, compare
├── 11-open-source-rag-stack/
│   ├── foundations.ipynb          # LangChain vs LlamaIndex vs Haystack for RAG — comparison
│   ├── advanced.ipynb            # Chroma vs Qdrant vs Milvus internals, benchmarking
│   ├── expert.ipynb              # building a framework-agnostic RAG layer, swappable components
│   └── build.ipynb               # build same RAG system in 3 frameworks, benchmark all
├── 12-production-rag-operations/
│   ├── foundations.ipynb          # failure modes, debugging retrieval quality
│   ├── advanced.ipynb            # handling updates/deletes, freshness, incremental indexing
│   ├── expert.ipynb              # scaling to millions of docs, multi-tenant RAG, operational runbooks
│   └── build.ipynb               # build production RAG with update pipeline + monitoring
└── capstone/
    └── production-rag-system/    # multi-source + graph + multimodal + eval + LangSmith monitoring
```

---

## Section 05: LangChain Ecosystem

Deep mastery of LangChain, LangGraph, and LangSmith — the tools you'll use to build most agent and RAG systems.

```
05-langchain-ecosystem/
├── 01-langchain-core/
│   ├── foundations.ipynb          # Runnables, LCEL, invoke/batch/stream
│   ├── advanced.ipynb            # RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch
│   ├── expert.ipynb              # custom Runnables, middleware patterns, lifecycle hooks
│   └── build.ipynb               # build complex chains with branching + parallel + fallbacks
├── 02-langchain-components/
│   ├── foundations.ipynb          # ChatModels, prompts, output parsers, document loaders
│   ├── advanced.ipynb            # custom LLM wrappers, caching, rate limiting, retry logic
│   ├── expert.ipynb              # custom document loaders, transformers, serialization
│   └── build.ipynb               # build a multi-provider gateway with caching + fallbacks
├── 03-langchain-tools/
│   ├── foundations.ipynb          # @tool decorator, StructuredTool, ToolMessage
│   ├── advanced.ipynb            # dynamic tool creation, tool error handling, toolkits
│   ├── expert.ipynb              # tool validation, injection prevention, tool composition
│   └── build.ipynb               # build a secure dynamic toolkit with validation
├── 04-langchain-retrieval/
│   ├── foundations.ipynb          # retrievers, vector store integration, document chains
│   ├── advanced.ipynb            # multi-vector, parent-document, ensemble, contextual compression
│   ├── expert.ipynb              # custom retrievers, self-querying, time-weighted
│   └── build.ipynb               # build a production retriever with multiple strategies
├── 05-langgraph-fundamentals/
│   ├── foundations.ipynb          # StateGraph, nodes, edges, conditional edges, state schema
│   ├── advanced.ipynb            # reducers, Annotated types, MessagesState, custom state
│   ├── expert.ipynb              # state design patterns, when to split state, shared vs private
│   └── build.ipynb               # build a stateful chatbot with well-designed state
├── 06-langgraph-control-flow/
│   ├── foundations.ipynb          # conditional routing, branching, cycles, recursion limits
│   ├── advanced.ipynb            # map-reduce with Send API, parallel node execution
│   ├── expert.ipynb              # dynamic graph construction, runtime-configurable topologies
│   └── build.ipynb               # build a dynamic workflow engine
├── 07-langgraph-persistence/
│   ├── foundations.ipynb          # checkpointing, MemorySaver, thread_id, thread_ts
│   ├── advanced.ipynb            # PostgresSaver, custom checkpointers, state recovery
│   ├── expert.ipynb              # time-travel debugging, checkpoint inspection, state migration
│   └── build.ipynb               # build persistent agent with time-travel capabilities
├── 08-langgraph-human-in-the-loop/
│   ├── foundations.ipynb          # interrupt(), Command, resume, basic approval flows
│   ├── advanced.ipynb            # edit state mid-run, multi-step review, reject + redirect
│   ├── expert.ipynb              # complex approval DAGs, escalation, timeout handling
│   └── build.ipynb               # build multi-stage approval agent with escalation
├── 09-langgraph-streaming/
│   ├── foundations.ipynb          # stream_mode (values, updates, messages), basic streaming
│   ├── advanced.ipynb            # astream_events, token streaming, custom events
│   ├── expert.ipynb              # streaming from subgraphs, multiplexed streams, backpressure
│   └── build.ipynb               # build a streaming chat UI with real-time events
├── 10-langgraph-subgraphs/
│   ├── foundations.ipynb          # subgraph nodes, state mapping between graphs
│   ├── advanced.ipynb            # nested subgraphs, cross-graph communication
│   ├── expert.ipynb              # hierarchical agent architectures, shared state strategies
│   └── build.ipynb               # build supervisor orchestrating specialized subgraphs
├── 11-langgraph-memory/
│   ├── foundations.ipynb          # Store API, namespaces, memory schemas
│   ├── advanced.ipynb            # semantic memory, cross-thread memory, memory indexing
│   ├── expert.ipynb              # memory architectures: episodic, semantic, procedural
│   └── build.ipynb               # build an agent with long-term memory across sessions
├── 12-langgraph-advanced-patterns/
│   ├── foundations.ipynb          # Command pattern, goto, graph updates
│   ├── advanced.ipynb            # dynamic node spawning, fan-out/fan-in, retry patterns
│   ├── expert.ipynb              # building custom LangGraph primitives, extending the framework
│   └── build.ipynb               # build a research swarm with dynamic agent spawning
├── 13-langsmith/
│   ├── foundations.ipynb          # tracing, runs, projects, annotation queues
│   ├── advanced.ipynb            # datasets, evaluators, pairwise eval, few-shot from traces
│   ├── expert.ipynb              # custom evaluators, online eval, automations, rules engine
│   └── build.ipynb               # build full eval pipeline: dataset → eval → comparison → CI
├── 14-langgraph-platform/
│   ├── foundations.ipynb          # LangGraph Server, assistants API, deployments
│   ├── advanced.ipynb            # cron tasks, double-texting, webhooks, background runs
│   ├── expert.ipynb              # custom auth, multi-tenant, horizontal scaling
│   └── build.ipynb               # deploy multi-agent system to LangGraph Platform
├── 15-open-source-observability/
│   ├── foundations.ipynb          # LangFuse setup, tracing, sessions, scoring
│   ├── advanced.ipynb            # Arize Phoenix, trace analysis, embeddings visualization
│   ├── expert.ipynb              # Helicone, proxy-based logging, cost tracking, comparing platforms
│   └── build.ipynb               # instrument same app with LangFuse + Phoenix + LangSmith, compare
└── capstone/
    └── full-stack-langgraph/     # multi-agent + HITL + streaming + persistence + eval + deployed
```

---

## Section 06: Agent Frameworks

Tiered mastery: expert depth on the top 3, working knowledge of the rest, and the meta-skill of knowing when to use what.

```
06-agent-frameworks/

  # TIER 1: Expert depth (foundations → advanced → expert → build)

├── 01-langgraph/
│   └── (covered in section 05 — cross-reference, no duplication)
├── 02-openai-agents-sdk/
│   ├── foundations.ipynb          # Agent class, Runner, handoffs, tools
│   ├── advanced.ipynb            # guardrails, tracing, context management, streaming
│   ├── expert.ipynb              # custom guardrails, multi-agent orchestration, production patterns
│   └── build.ipynb               # build a multi-agent system with handoffs + guardrails
├── 03-claude-agent-sdk/
│   ├── foundations.ipynb          # agent loops, tool use, system prompts
│   ├── advanced.ipynb            # extended thinking, computer use, multi-turn tool use
│   ├── expert.ipynb              # custom tool implementations, orchestration, production patterns
│   └── build.ipynb               # build an agent system leveraging extended thinking + computer use

  # TIER 2: Working knowledge (foundations → applied → build)

├── 04-crewai/
│   ├── foundations.ipynb          # agents, tasks, crews, roles, process types
│   ├── applied.ipynb             # custom tools, memory, delegation, async crews
│   └── build.ipynb               # build a research crew with specialized agents
├── 05-autogen/
│   ├── foundations.ipynb          # conversable agents, group chat, AG2
│   ├── applied.ipynb             # nested chats, teachability, Magentic-One
│   └── build.ipynb               # build a collaborative agent team
├── 06-dspy/
│   ├── foundations.ipynb          # signatures, modules, optimizers
│   ├── applied.ipynb             # assertions, typed predictors, compilation
│   └── build.ipynb               # build an optimized RAG pipeline with DSPy
├── 07-semantic-kernel/
│   ├── foundations.ipynb          # kernel, plugins, planners
│   ├── applied.ipynb             # auto function calling, Azure AI integration
│   └── build.ipynb               # build enterprise agent with Semantic Kernel
├── 08-haystack/
│   ├── foundations.ipynb          # pipelines, components, document stores
│   ├── applied.ipynb             # branching pipelines, agents, Haystack 2.0
│   └── build.ipynb               # build RAG + agent system with Haystack
├── 09-llamaindex-agents/
│   ├── foundations.ipynb          # agent framework, tools, query planning
│   ├── applied.ipynb             # workflows, event-driven agents, LlamaDeploy
│   └── build.ipynb               # build agent system with LlamaIndex Workflows

  # TIER 3: The meta-skill

├── 10-mcp/
│   ├── foundations.ipynb          # Model Context Protocol spec, servers, clients, transports
│   ├── advanced.ipynb            # building MCP servers, resources, prompts, tool schemas
│   ├── expert.ipynb              # multi-server orchestration, auth, security, custom transports
│   └── build.ipynb               # build production MCP servers and integrate across frameworks
├── 11-framework-selection/
│   ├── decision-matrix.ipynb     # when to use what — complexity, team size, use case mapping
│   ├── migration-patterns.ipynb  # moving between frameworks, abstraction layers
│   ├── benchmarks.ipynb          # same agent in LangGraph vs OpenAI vs Claude SDK — compare
│   └── build.ipynb               # build a framework-agnostic agent abstraction layer
└── capstone/
    └── same-agent-three-ways/    # identical agent in LangGraph + OpenAI SDK + Claude SDK, with eval
```

---

## Section 07: Agent Architectures

Framework-agnostic patterns and architectures for building intelligent agents. The conceptual depth that makes you an agent architect.

```
07-agent-architectures/
├── 01-single-agent-patterns/
│   ├── foundations.ipynb          # ReAct, function calling, tool-use loop
│   ├── advanced.ipynb            # reflexion, self-critique, chain-of-thought agents
│   ├── expert.ipynb              # LATS (Language Agent Tree Search), Monte Carlo agents
│   └── build.ipynb               # implement ReAct, reflexion, and LATS from scratch
├── 02-planning-agents/
│   ├── foundations.ipynb          # plan-and-execute, task decomposition
│   ├── advanced.ipynb            # hierarchical planning, plan refinement, re-planning
│   ├── expert.ipynb              # LLM-Modulo framework, verifier-guided planning
│   └── build.ipynb               # build a planner with verification and re-planning
├── 03-multi-agent-systems/
│   ├── foundations.ipynb          # supervisor, hierarchical, peer-to-peer topologies
│   ├── advanced.ipynb            # debate, voting, consensus, specialization
│   ├── expert.ipynb              # emergent behavior, communication protocols, game theory
│   └── build.ipynb               # build a debate-based multi-agent decision system
├── 04-memory-architectures/
│   ├── foundations.ipynb          # buffer, window, summary, vector memory
│   ├── advanced.ipynb            # episodic, semantic, procedural memory systems
│   ├── expert.ipynb              # MemGPT/Letta, memory consolidation, forgetting strategies
│   └── build.ipynb               # build a MemGPT-style tiered memory system
├── 05-coding-agents/
│   ├── foundations.ipynb          # code generation, execution, testing loops
│   ├── advanced.ipynb            # SWE-agent patterns, repository-level reasoning
│   ├── expert.ipynb              # self-debugging, program repair, sandboxed execution
│   └── build.ipynb               # build a coding agent that writes, tests, and debugs code
├── 06-research-and-search-agents/
│   ├── foundations.ipynb          # web search, Tavily/Exa integration, information synthesis
│   ├── advanced.ipynb            # deep research patterns, iterative refinement, citation
│   ├── expert.ipynb              # multi-source verification, claim decomposition, AI search (Perplexity-style)
│   └── build.ipynb               # build a deep research agent with source verification and citations
├── 07-text-to-sql-agents/
│   ├── foundations.ipynb          # NL→SQL basics, schema understanding, query generation
│   ├── advanced.ipynb            # complex joins, multi-step queries, result visualization
│   ├── expert.ipynb              # safety (read-only, query validation), multi-database, pandas-ai patterns
│   └── build.ipynb               # build a data analyst agent: NL → SQL → visualization
├── 08-agent-evaluation/
│   ├── foundations.ipynb          # task completion, tool accuracy, trajectory eval
│   ├── advanced.ipynb            # SWE-bench, AgentBench, GAIA, WebArena
│   ├── expert.ipynb              # statistical evaluation, confidence intervals, A/B testing agents
│   └── build.ipynb               # build an agent evaluation framework
├── 09-agent-safety/
│   ├── foundations.ipynb          # guardrails, sandboxing, permission systems
│   ├── advanced.ipynb            # prompt injection defense, tool abuse prevention
│   ├── expert.ipynb              # formal verification, constitutional agents, alignment
│   └── build.ipynb               # build a safety layer with guardrails + monitoring
└── capstone/
    └── autonomous-agent-system/  # planning + memory + safety + eval, fully autonomous
```

---

## Section 08: Vision

```
08-vision/
├── 01-convnets/
│   ├── foundations.ipynb          # convolutions, pooling, feature maps — implement from scratch
│   ├── advanced.ipynb            # ResNet, EfficientNet, ConvNeXt, architecture evolution
│   ├── expert.ipynb              # neural architecture search, depthwise separable, inverted residuals
│   └── build.ipynb               # build and train CNN from scratch, compare to pretrained
├── 02-vision-transformers/
│   ├── foundations.ipynb          # ViT, patch embeddings, position encoding
│   ├── advanced.ipynb            # DINOv2, MAE, Swin, BEiT, self-supervised
│   ├── expert.ipynb              # scaling ViTs, hybrid architectures, efficiency
│   └── build.ipynb               # implement ViT from scratch, train with self-supervised
├── 03-object-detection/
│   ├── foundations.ipynb          # anchor boxes, IoU, NMS, YOLO architecture
│   ├── advanced.ipynb            # YOLOv8/v10, RT-DETR, end-to-end detection
│   ├── expert.ipynb              # Grounding DINO, open-vocab detection, few-shot detection
│   └── build.ipynb               # train and deploy detector with open-vocab capability
├── 04-segmentation/
│   ├── foundations.ipynb          # semantic, instance, panoptic segmentation
│   ├── advanced.ipynb            # SAM, SAM2, Mask R-CNN
│   ├── expert.ipynb              # open-vocab segmentation, video segmentation, interactive
│   └── build.ipynb               # build interactive segmentation pipeline with SAM2
├── 05-image-generation/
│   ├── foundations.ipynb          # diffusion process, noise schedules, U-Net, VAE
│   ├── advanced.ipynb            # Stable Diffusion, SDXL, Flux, ControlNet
│   ├── expert.ipynb              # LoRA, IP-Adapter, textual inversion, consistency models
│   └── build.ipynb               # fine-tune SDXL with LoRA, deploy with ControlNet
├── 06-3d-vision/
│   ├── foundations.ipynb          # depth estimation, point clouds, camera geometry
│   ├── advanced.ipynb            # NeRF, Gaussian splatting, multi-view synthesis
│   ├── expert.ipynb              # 3D generation from images, reconstruction at scale
│   └── build.ipynb               # build a 3D reconstruction pipeline
└── capstone/
    └── vision-ai-platform/       # detection + segmentation + generation + 3D pipeline
```

---

## Section 09: Audio

```
09-audio/
├── 01-audio-fundamentals/
│   ├── foundations.ipynb          # waveforms, spectrograms, mel-frequency, MFCCs
│   ├── advanced.ipynb            # audio augmentation, feature engineering, codecs
│   ├── expert.ipynb              # neural audio codecs (EnCodec, DAC), learned representations
│   └── build.ipynb               # build audio feature extraction + codec pipeline
├── 02-speech-recognition/
│   ├── foundations.ipynb          # CTC, attention-based ASR, Whisper architecture
│   ├── advanced.ipynb            # fine-tuning Whisper, streaming ASR, word timestamps
│   ├── expert.ipynb              # multi-language, code-switching, noise robustness, distillation
│   └── build.ipynb               # build production ASR with streaming + diarization
├── 03-speaker-analysis/
│   ├── foundations.ipynb          # speaker embeddings, verification, identification
│   ├── advanced.ipynb            # diarization, speaker change detection, clustering
│   ├── expert.ipynb              # multi-speaker scenarios, overlap handling
│   └── build.ipynb               # build speaker diarization system
├── 04-text-to-speech/
│   ├── foundations.ipynb          # mel spectrograms, vocoders, autoregressive TTS
│   ├── advanced.ipynb            # VITS, Bark, voice cloning, zero-shot TTS
│   ├── expert.ipynb              # emotion/style control, prosody, real-time streaming TTS
│   └── build.ipynb               # build TTS with voice cloning and emotion control
├── 05-music-and-audio-generation/
│   ├── foundations.ipynb          # MIDI, symbolic music, audio language models
│   ├── advanced.ipynb            # MusicGen, AudioCraft, Stable Audio
│   ├── expert.ipynb              # controllable generation, inpainting, source separation
│   └── build.ipynb               # build music generation + manipulation pipeline
└── capstone/
    └── voice-ai-assistant/       # ASR + LLM + TTS + speaker ID, real-time voice agent
```

---

## Section 10: Video

```
10-video/
├── 01-video-fundamentals/
│   ├── foundations.ipynb          # temporal modeling, optical flow, frame sampling
│   ├── advanced.ipynb            # video transformers, TimeSformer, ViViT, VideoMAE
│   ├── expert.ipynb              # long-video understanding, temporal attention patterns
│   └── build.ipynb               # build video feature extraction pipeline
├── 02-video-understanding/
│   ├── foundations.ipynb          # action recognition, classification, captioning
│   ├── advanced.ipynb            # temporal grounding, video QA, moment retrieval
│   ├── expert.ipynb              # video-language models, dense video captioning
│   └── build.ipynb               # build video understanding system with QA
├── 03-tracking/
│   ├── foundations.ipynb          # MOT, SORT, DeepSORT, tracking-by-detection
│   ├── advanced.ipynb            # ByteTrack, BoT-SORT, SAM2 video tracking
│   ├── expert.ipynb              # point tracking (CoTracker), long-term tracking, occlusion
│   └── build.ipynb               # build real-time multi-object tracker with re-ID
├── 04-video-generation/
│   ├── foundations.ipynb          # video diffusion, temporal consistency, motion
│   ├── advanced.ipynb            # SVD, AnimateDiff, Sora-like architectures
│   ├── expert.ipynb              # controllable generation, video editing, style transfer
│   └── build.ipynb               # build text-to-video generation pipeline
└── capstone/
    └── video-intelligence/       # understanding + tracking + generation end-to-end
```

---

## Section 11: Multimodal

```
11-multimodal/
├── 01-contrastive-models/
│   ├── foundations.ipynb          # CLIP architecture, contrastive learning, zero-shot
│   ├── advanced.ipynb            # SigLIP, EVA-CLIP, fine-tuning CLIP
│   ├── expert.ipynb              # CLAP (audio), ImageBind, unified embedding spaces
│   └── build.ipynb               # build multi-modal search engine with CLIP + CLAP
├── 02-vision-language-models/
│   ├── foundations.ipynb          # LLaVA architecture, visual instruction tuning
│   ├── advanced.ipynb            # Qwen-VL, InternVL, GPT-4V patterns, Gemini native
│   ├── expert.ipynb              # training VLMs, data recipes, scaling, grounding
│   └── build.ipynb               # fine-tune a VLM on custom visual tasks
├── 03-document-ai/
│   ├── foundations.ipynb          # OCR, layout analysis, table extraction
│   ├── advanced.ipynb            # DocTR, LayoutLM, multi-page understanding
│   ├── expert.ipynb              # ColPali, visually-rich document understanding
│   └── build.ipynb               # build document parsing pipeline for complex PDFs
├── 04-any-to-any-models/
│   ├── foundations.ipynb          # unified multimodal architectures, tokenization strategies
│   ├── advanced.ipynb            # audio-visual models, cross-modal generation
│   ├── expert.ipynb              # Chameleon, unified autoregressive multimodal models
│   └── build.ipynb               # build multimodal pipeline combining all modalities
└── capstone/
    └── multimodal-ai-assistant/  # text + image + audio + document, any-to-any
```

---

## Section 12: Fine-Tuning

```
12-fine-tuning/
├── 01-data-engineering/
│   ├── foundations.ipynb          # data collection, cleaning, formatting
│   ├── advanced.ipynb            # quality filtering, deduplication, data mixing
│   ├── expert.ipynb              # data flywheel, active learning, curriculum design
│   └── build.ipynb               # build a data curation pipeline with quality scoring
├── 02-parameter-efficient/
│   ├── foundations.ipynb          # LoRA, QLoRA, adapters, prefix tuning
│   ├── advanced.ipynb            # DoRA, LongLoRA, multi-task adapters, adapter merging
│   ├── expert.ipynb              # adapter composition, weight interpolation, model soups
│   └── build.ipynb               # fine-tune with QLoRA, merge, evaluate
├── 03-full-fine-tuning/
│   ├── foundations.ipynb          # setup, hyperparameters, learning rate schedules
│   ├── advanced.ipynb            # multi-GPU, FSDP, gradient checkpointing
│   ├── expert.ipynb              # curriculum learning, continual pretraining, domain adaptation
│   └── build.ipynb               # full fine-tune on custom instruction dataset
├── 04-vision-and-multimodal-ft/
│   ├── foundations.ipynb          # fine-tuning ViT, detection models, VLMs
│   ├── advanced.ipynb            # multi-task fine-tuning, adapting diffusion models
│   ├── expert.ipynb              # cross-modal transfer, modality-specific strategies
│   └── build.ipynb               # fine-tune VLM for domain-specific visual understanding
├── 05-evaluation/
│   ├── foundations.ipynb          # perplexity, task metrics, benchmark suites
│   ├── advanced.ipynb            # LM Eval Harness, human eval, contamination detection
│   ├── expert.ipynb              # custom benchmarks, statistical significance, Elo ratings
│   └── build.ipynb               # build evaluation suite for fine-tuned models
├── 06-open-source-ft-tools/
│   ├── foundations.ipynb          # Unsloth — 2x faster LoRA, setup and usage
│   ├── advanced.ipynb            # Axolotl — config-driven fine-tuning, multi-dataset
│   ├── expert.ipynb              # LitGPT, torchtune, choosing the right tool
│   └── build.ipynb               # same fine-tune with Unsloth vs Axolotl vs TRL — compare
├── 07-data-curation/
│   ├── foundations.ipynb          # Label Studio, Argilla — annotation workflows
│   ├── advanced.ipynb            # active learning loops, LLM-assisted labeling
│   ├── expert.ipynb              # data flywheel: production feedback → labeling → retraining
│   └── build.ipynb               # build annotation pipeline with Argilla + synthetic data
├── 08-synthetic-data/
│   ├── foundations.ipynb          # using LLMs to generate training data, Self-Instruct
│   ├── advanced.ipynb            # Evol-Instruct, distillation pipelines, data augmentation
│   ├── expert.ipynb              # quality filtering, decontamination, domain-specific synthetic data
│   └── build.ipynb               # build synthetic data generation + filtering pipeline
├── 09-model-distillation/
│   ├── foundations.ipynb          # knowledge distillation, teacher-student framework
│   ├── advanced.ipynb            # distilling GPT-4/Claude into smaller models for cost
│   ├── expert.ipynb              # quality-cost Pareto optimization, when distillation beats fine-tuning
│   └── build.ipynb               # build distillation pipeline: large → small with eval
└── capstone/
    └── domain-expert-model/      # curate → train → evaluate → deploy → monitor
```

---

## Section 13: Evaluation

A dedicated discipline. The ability to rigorously evaluate AI systems is what separates production engineers from prototype builders.

```
13-evaluation/
├── 01-llm-evaluation/
│   ├── foundations.ipynb          # metrics, benchmarks, MMLU, HumanEval, GPQA
│   ├── advanced.ipynb            # LLM-as-judge, pairwise comparison, Chatbot Arena
│   ├── expert.ipynb              # contamination, benchmark gaming, designing robust evals
│   └── build.ipynb               # build LLM-as-judge eval system with calibration
├── 02-rag-evaluation/
│   ├── foundations.ipynb          # RAGAS, component-wise metrics
│   ├── advanced.ipynb            # end-to-end eval, synthetic test data, regression testing
│   ├── expert.ipynb              # custom metrics, attribution evaluation, factual consistency
│   └── build.ipynb               # build comprehensive RAG eval pipeline
├── 03-agent-evaluation/
│   ├── foundations.ipynb          # task completion, trajectory eval, tool accuracy
│   ├── advanced.ipynb            # SWE-bench, AgentBench, GAIA, WebArena
│   ├── expert.ipynb              # multi-turn eval, safety testing, adversarial testing
│   └── build.ipynb               # build agent evaluation harness
├── 04-safety-evaluation/
│   ├── foundations.ipynb          # toxicity, bias, fairness metrics
│   ├── advanced.ipynb            # red-teaming, jailbreak testing, robustness
│   ├── expert.ipynb              # automated red-teaming, constitutional evaluation
│   └── build.ipynb               # build safety evaluation + red-teaming pipeline
├── 05-production-evaluation/
│   ├── foundations.ipynb          # online metrics, user feedback, A/B testing
│   ├── advanced.ipynb            # LangSmith production eval, drift detection
│   ├── expert.ipynb              # continuous evaluation, guardrail monitoring, quality gates
│   └── build.ipynb               # build production eval system with LangSmith
├── 06-open-source-eval/
│   ├── foundations.ipynb          # DeepEval, RAGAS, Promptfoo — setup and comparison
│   ├── advanced.ipynb            # custom evaluators, CI/CD integration, eval-driven development
│   ├── expert.ipynb              # building eval datasets, statistical rigor, eval contamination
│   └── build.ipynb               # build eval pipeline: DeepEval + Promptfoo in CI/CD
└── capstone/
    └── eval-platform/            # unified eval: LLM + RAG + agent + safety + production
```

---

## Section 14: Safety and Alignment

```
14-safety-and-alignment/
├── 01-prompt-security/
│   ├── foundations.ipynb          # prompt injection, jailbreaks, common attack vectors
│   ├── advanced.ipynb            # indirect injection via RAG, multi-turn attacks, data exfiltration
│   ├── expert.ipynb              # defense in depth, input/output classifiers, canary tokens
│   └── build.ipynb               # build a prompt security layer with multi-stage defense
├── 02-guardrails/
│   ├── foundations.ipynb          # content filtering, topic control, output validation
│   ├── advanced.ipynb            # NeMo Guardrails, Guardrails AI, LLM-based guards
│   ├── expert.ipynb              # custom guardrail pipelines, latency vs safety tradeoffs
│   └── build.ipynb               # build a production guardrails system
├── 03-alignment/
│   ├── foundations.ipynb          # alignment problem, reward hacking, specification gaming
│   ├── advanced.ipynb            # RLHF failures, reward model evaluation, scalable oversight
│   ├── expert.ipynb              # constitutional AI in practice, debate, recursive reward modeling
│   └── build.ipynb               # implement constitutional AI pipeline
├── 04-responsible-ai/
│   ├── foundations.ipynb          # bias detection, fairness metrics, model cards
│   ├── advanced.ipynb            # debiasing techniques, representation analysis
│   ├── expert.ipynb              # regulatory compliance (EU AI Act), audit trails, documentation
│   └── build.ipynb               # build responsible AI audit pipeline
├── 05-auth-and-compliance/
│   ├── foundations.ipynb          # API key management, rate limiting, basic access control
│   ├── advanced.ipynb            # multi-tenancy, per-tenant data isolation in RAG, row-level security
│   ├── expert.ipynb              # PII detection/redaction, GDPR/SOC2/HIPAA for AI systems, audit logging
│   └── build.ipynb               # build a compliant AI system with PII redaction + audit trails
└── capstone/
    └── safe-ai-system/           # security + guardrails + alignment + compliance + monitoring
```

---

## Section 15: AI System Design

The staff-level differentiator. Not how things work, but how to architect AI systems that work in production.

```
15-ai-system-design/
├── 01-design-patterns/
│   ├── foundations.ipynb          # RAG vs fine-tuning vs long context — decision framework
│   ├── advanced.ipynb            # routing, cascading, fallback architectures
│   ├── expert.ipynb              # multi-model orchestration, hybrid architectures
│   └── build.ipynb               # design + implement a model router with cost optimization
├── 02-cost-engineering/
│   ├── foundations.ipynb          # token economics, pricing models, budget estimation
│   ├── advanced.ipynb            # semantic caching, prompt caching, KV cache sharing
│   ├── expert.ipynb              # cost-aware routing, batching strategies, capacity planning
│   └── build.ipynb               # build a cost-aware AI gateway with caching
├── 03-caching-patterns/
│   ├── foundations.ipynb          # exact-match caching, API-level prompt caching
│   ├── advanced.ipynb            # semantic caching (GPTCache), embedding caching
│   ├── expert.ipynb              # cache invalidation, staleness tradeoffs, when caching hurts
│   └── build.ipynb               # build multi-layer caching system for AI application
├── 04-latency-optimization/
│   ├── foundations.ipynb          # streaming, async, parallel tool calls
│   ├── advanced.ipynb            # speculative execution, pre-computation, edge caching
│   ├── expert.ipynb              # P99 optimization, tail latency, SLA design
│   └── build.ipynb               # build a low-latency serving pipeline with SLA monitoring
├── 05-error-handling-and-resilience/
│   ├── foundations.ipynb          # retry strategies, timeout handling, graceful degradation
│   ├── advanced.ipynb            # circuit breakers, fallback model chains, idempotency
│   ├── expert.ipynb              # partial failure in multi-agent workflows, self-healing systems
│   └── build.ipynb               # build resilient AI pipeline with circuit breakers + fallbacks
├── 06-system-architectures/
│   ├── foundations.ipynb          # chatbot, copilot, autonomous agent — architecture templates
│   ├── advanced.ipynb            # multi-tenant AI, platform design, plugin systems
│   ├── expert.ipynb              # designing for scale: 10k→100k→1M users, data isolation
│   └── build.ipynb               # design and build a multi-tenant AI platform
├── 07-cloud-platforms/
│   ├── foundations.ipynb          # AWS Bedrock, Azure AI Studio, GCP Vertex AI
│   ├── advanced.ipynb            # Modal, RunPod, Together AI, Replicate — serverless GPU
│   ├── expert.ipynb              # SkyPilot, multi-cloud, cost optimization across providers
│   └── build.ipynb               # deploy same system across 3 cloud providers, compare
├── 08-ui-and-ux/
│   ├── foundations.ipynb          # Gradio, Streamlit for AI apps
│   ├── advanced.ipynb            # Chainlit for chat, Next.js + Vercel AI SDK patterns
│   ├── expert.ipynb              # streaming UX, progressive disclosure, error states, feedback loops
│   └── build.ipynb               # build production chat UI with streaming + feedback collection
├── 09-case-studies/
│   ├── customer-support.ipynb    # design: multilingual support system with escalation
│   ├── code-assistant.ipynb      # design: repository-aware coding agent
│   ├── data-analyst.ipynb        # design: natural language → SQL → visualization
│   ├── content-pipeline.ipynb    # design: multimodal content generation at scale
│   └── knowledge-base.ipynb      # design: enterprise RAG over heterogeneous sources
└── capstone/
    └── system-design-portfolio/  # 3 complete system designs with architecture docs + prototypes
```

---

## Section 16: Data Engineering for AI

The unglamorous but critical work of getting data into AI systems.

```
16-data-engineering-for-ai/
├── 01-document-parsing/
│   ├── foundations.ipynb          # PDF, DOCX, HTML extraction basics
│   ├── advanced.ipynb            # Unstructured.io, LlamaParse, Docling, table extraction
│   ├── expert.ipynb              # multi-format pipelines, quality scoring, handling messy data
│   └── build.ipynb               # build production document ingestion pipeline
├── 02-data-pipelines/
│   ├── foundations.ipynb          # ETL for knowledge bases, scheduling, incremental updates
│   ├── advanced.ipynb            # streaming ingestion, CDC, data versioning (DVC)
│   ├── expert.ipynb              # scaling to millions of docs, deduplication, quality monitoring
│   └── build.ipynb               # build scalable RAG ingestion pipeline with monitoring
├── 03-web-data/
│   ├── foundations.ipynb          # web scraping, crawling, API data collection
│   ├── advanced.ipynb            # LLM-powered extraction, structured scraping, rate limiting
│   ├── expert.ipynb              # legal/ethical considerations, robots.txt, responsible crawlers
│   └── build.ipynb               # build a web knowledge collector with structured extraction
└── capstone/
    └── data-platform/            # multi-source ingestion + quality + versioning + monitoring
```

---

## Section 17: Testing and CI for AI

Software engineering discipline applied to AI systems. What most data scientists lack.

```
17-testing-and-ci-for-ai/
├── 01-unit-testing/
│   ├── foundations.ipynb          # testing prompt templates, output parsers, chains
│   ├── advanced.ipynb            # mocking LLM calls, deterministic testing, seeded generation
│   ├── expert.ipynb              # snapshot testing, property-based testing for AI outputs
│   └── build.ipynb               # build a test suite for an LLM application
├── 02-integration-testing/
│   ├── foundations.ipynb          # testing RAG pipelines end-to-end, agent workflows
│   ├── advanced.ipynb            # testing with real models vs mocks, cost-aware test strategies
│   ├── expert.ipynb              # testing multi-agent systems, stateful workflow testing
│   └── build.ipynb               # build integration test suite for agent system
├── 03-ci-cd-for-ai/
│   ├── foundations.ipynb          # GitHub Actions for AI, running evals in CI, Promptfoo in CI
│   ├── advanced.ipynb            # regression testing prompt changes, model upgrade testing
│   ├── expert.ipynb              # canary deployments, blue-green for models, rollback strategies
│   └── build.ipynb               # build full CI/CD pipeline for LLM application
├── 04-load-and-performance/
│   ├── foundations.ipynb          # load testing inference endpoints, Locust, k6
│   ├── advanced.ipynb            # throughput vs latency tradeoffs, batching under load
│   ├── expert.ipynb              # capacity planning, autoscaling triggers, SLA validation
│   └── build.ipynb               # build performance testing + capacity planning suite
└── capstone/
    └── tested-ai-system/         # fully tested + CI/CD + performance validated AI app
```

---

## Section 18: Workflow Orchestration

Durable, event-driven, and scheduled AI workflows beyond LangGraph.

```
18-workflow-orchestration/
├── 01-temporal/
│   ├── foundations.ipynb          # workflows, activities, durable execution basics
│   ├── advanced.ipynb            # long-running AI workflows, human-in-the-loop with Temporal
│   ├── expert.ipynb              # retry policies, saga patterns, versioning workflows
│   └── build.ipynb               # build a durable multi-step agent workflow with Temporal
├── 02-event-driven-ai/
│   ├── foundations.ipynb          # event-driven architecture, message queues, Kafka basics
│   ├── advanced.ipynb            # Inngest, event-triggered AI pipelines, webhook-driven agents
│   ├── expert.ipynb              # complex event processing, real-time decision systems
│   └── build.ipynb               # build event-driven AI pipeline
├── 03-pipeline-orchestration/
│   ├── foundations.ipynb          # Prefect, Airflow for AI/ML pipelines
│   ├── advanced.ipynb            # DAGs for training + eval + deploy, scheduled retraining
│   ├── expert.ipynb              # multi-pipeline coordination, data + model + eval pipelines
│   └── build.ipynb               # build end-to-end ML pipeline with orchestration
└── capstone/
    └── orchestrated-ai-platform/ # durable agents + event-driven + scheduled pipelines
```

---

## Section 19: Real-Time AI

Where the industry is heading. Streaming, voice, and live multimodal interaction.

```
19-real-time-ai/
├── 01-streaming-fundamentals/
│   ├── foundations.ipynb          # SSE, WebSockets, streaming protocols
│   ├── advanced.ipynb            # token streaming, astream_events, backpressure
│   ├── expert.ipynb              # multiplexed streams, real-time orchestration
│   └── build.ipynb               # build streaming AI backend with WebSocket + SSE
├── 02-voice-pipelines/
│   ├── foundations.ipynb          # ASR → LLM → TTS pipeline, latency chain
│   ├── advanced.ipynb            # OpenAI Realtime API, Gemini Live, interruption handling
│   ├── expert.ipynb              # voice activity detection, turn-taking, emotion-aware
│   └── build.ipynb               # build real-time voice agent with interruption support
├── 03-real-time-vision/
│   ├── foundations.ipynb          # video stream processing, frame-rate inference
│   ├── advanced.ipynb            # real-time detection, edge inference, camera pipelines
│   ├── expert.ipynb              # multi-camera systems, latency budgets, GPU scheduling
│   └── build.ipynb               # build real-time video analysis pipeline
├── 04-live-multimodal/
│   ├── foundations.ipynb          # screen sharing, live audio+video, co-browsing
│   ├── advanced.ipynb            # multimodal fusion in real-time, shared context
│   ├── expert.ipynb              # Project Astra-style architectures, always-on AI
│   └── build.ipynb               # build a live multimodal assistant
└── capstone/
    └── real-time-ai-assistant/   # voice + vision + screen, real-time multimodal agent
```

---

## Section 20: Edge and On-Device

Running AI where there's no cloud.

```
20-edge-and-on-device/
├── 01-model-compression/
│   ├── foundations.ipynb          # quantization (int8, int4), pruning, distillation
│   ├── advanced.ipynb            # GGUF, CoreML, TFLite conversion pipelines
│   ├── expert.ipynb              # neural architecture search for edge, hardware-aware NAS
│   └── build.ipynb               # compress and convert model for 3 target platforms
├── 02-on-device-inference/
│   ├── foundations.ipynb          # llama.cpp, MLX, ONNX Runtime
│   ├── advanced.ipynb            # WebLLM, browser inference, WebGPU
│   ├── expert.ipynb              # mobile deployment (CoreML, TFLite), NPU utilization
│   └── build.ipynb               # deploy LLM to browser + mobile + desktop
├── 03-hybrid-architectures/
│   ├── foundations.ipynb          # edge-cloud split, when to run locally vs API
│   ├── advanced.ipynb            # speculative local + cloud verification, offline-first
│   ├── expert.ipynb              # federated learning, on-device fine-tuning, privacy-preserving
│   └── build.ipynb               # build hybrid edge-cloud AI system
└── capstone/
    └── edge-ai-platform/         # on-device + cloud fallback + sync + privacy
```

---

## Section 21: Deployment and MLOps

Getting AI systems to production and keeping them running.

```
21-deployment-and-mlops/
├── 01-serving/
│   ├── foundations.ipynb          # REST APIs, FastAPI, model loading patterns
│   ├── advanced.ipynb            # vLLM, TGI, TensorRT-LLM, SGLang, batching strategies
│   ├── expert.ipynb              # multi-model serving, Triton, model routing, A/B serving
│   └── build.ipynb               # deploy LLM with vLLM + model router
├── 02-optimization/
│   ├── foundations.ipynb          # server-side optimization: batching, caching, compiled models
│   ├── advanced.ipynb            # ONNX, TensorRT, torch.compile, kernel fusion
│   ├── expert.ipynb              # custom CUDA kernels, Triton language, hardware-aware optimization
│   └── build.ipynb               # optimize serving pipeline end-to-end (distinct from edge compression in S20)
├── 03-infrastructure/
│   ├── foundations.ipynb          # Docker, GPU setup, cloud providers (AWS/GCP/Azure)
│   ├── advanced.ipynb            # Kubernetes, Ray Serve, autoscaling, BentoML, KServe
│   ├── expert.ipynb              # multi-region, edge deployment, cost optimization
│   └── build.ipynb               # containerize + deploy with GPU autoscaling
├── 04-mlops/
│   ├── foundations.ipynb          # experiment tracking (MLflow, W&B), model registry, CI/CD
│   ├── advanced.ipynb            # ML pipelines (Kubeflow, Airflow), feature stores
│   ├── expert.ipynb              # LLMOps vs MLOps, prompt management, config-driven systems
│   └── build.ipynb               # build full MLOps/LLMOps pipeline
├── 05-monitoring/
│   ├── foundations.ipynb          # logging, metrics, alerting, dashboards
│   ├── advanced.ipynb            # LangSmith + LangFuse monitoring, cost tracking, quality metrics
│   ├── expert.ipynb              # anomaly detection, drift detection, auto-remediation
│   └── build.ipynb               # build monitoring + alerting system
└── capstone/
    └── production-ai-platform/   # full stack: serve + optimize + scale + monitor + CI/CD
```

---

## Section 22: Research Paper Implementations

Implementing landmark papers from scratch to build deep understanding. Each notebook implements the core algorithm of the paper with clear annotations.

```
22-research-papers/
├── 01-attention-is-all-you-need.ipynb     # implement original transformer
├── 02-bert.ipynb                          # implement BERT pretraining
├── 03-gpt.ipynb                          # implement GPT architecture
├── 04-clip.ipynb                          # implement contrastive learning
├── 05-ddpm.ipynb                          # implement denoising diffusion
├── 06-lora.ipynb                          # implement LoRA from scratch
├── 07-flash-attention.ipynb               # implement flash attention
├── 08-rag-original.ipynb                  # implement original RAG paper
├── 09-react.ipynb                         # implement ReAct agent pattern
├── 10-self-rag.ipynb                      # implement Self-RAG
├── 11-dpo.ipynb                           # implement Direct Preference Optimization
├── 12-graph-rag.ipynb                     # implement GraphRAG
├── 13-mamba.ipynb                         # implement state space model
├── 14-mixture-of-experts.ipynb            # implement MoE routing
├── 15-vision-transformer.ipynb            # implement ViT from scratch
├── 16-swe-agent.ipynb                     # implement SWE-agent patterns
├── 17-toolformer.ipynb                    # implement Toolformer
├── 18-crag.ipynb                          # implement Corrective RAG
├── 19-raptor.ipynb                        # implement RAPTOR tree retrieval
└── 20-memgpt.ipynb                        # implement MemGPT memory architecture
```

---

## Final Numbers

| Metric | Count |
|--------|-------|
| Sections | 22 |
| Topics | ~130 |
| Notebooks | ~530+ |
| Capstone projects | 21 |
| Paper implementations | 20 |
| Agent frameworks deep-dived | 9 (3 expert, 6 working knowledge) |
| Depth levels per topic | 4 (foundations → advanced → expert → build) |

## Priority Build Order

**Phase 1:** Sections 01, 04, 05 (LLMs, RAG, LangChain ecosystem)
**Phase 2:** Sections 06, 07 (Agent frameworks, Agent architectures)
**Phase 3:** Sections 02, 03 (HF ecosystem, Foundation models)
**Phase 4:** Sections 13, 15, 17 (Evaluation, System design, Testing)
**Phase 5:** Sections 08-11 (Vision, Audio, Video, Multimodal)
**Phase 6:** Sections 12, 14, 16, 18-22 (Fine-tuning, Safety, Data eng, Orchestration, Real-time, Edge, Deployment, Papers)
