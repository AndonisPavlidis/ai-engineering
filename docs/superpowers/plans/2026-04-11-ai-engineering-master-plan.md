# AI Engineering Curriculum — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive, expert-depth AI engineering learning repository with ~530 Jupyter notebooks across 22 sections, each following a foundations → advanced → expert → build progression.

**Architecture:** Each section is a self-contained module with numbered topic directories. Every topic has 3-4 notebooks (foundations, advanced, expert, build) plus a capstone project per section. Notebooks are Jupyter (.ipynb) with markdown theory cells interleaved with runnable code. Section 01 is built first to establish the pattern, then sections proceed sequentially.

**Tech Stack:** Python 3.11+, Jupyter, PyTorch, Hugging Face Transformers, LangChain, LangGraph, LangSmith, and domain-specific libraries per section.

---

## Phase 0: Repository Scaffolding

### Task 0.1: Initialize repo structure

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `notebooks/template.ipynb`

- [ ] **Step 1: Create README.md**

```markdown
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
```

- [ ] **Step 2: Create requirements.txt with core dependencies**

```
# Core
jupyter>=1.0
jupyterlab>=4.0
ipywidgets>=8.0
matplotlib>=3.8
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4

# Deep Learning
torch>=2.2
torchvision>=0.17
torchaudio>=2.2

# Hugging Face
transformers>=4.40
datasets>=2.18
accelerate>=0.28
peft>=0.10
trl>=0.8
evaluate>=0.4
tokenizers>=0.15
sentence-transformers>=2.6

# LLM Providers
openai>=1.14
anthropic>=0.25
google-generativeai>=0.5

# LangChain Ecosystem
langchain>=0.2
langchain-core>=0.2
langchain-community>=0.2
langchain-openai>=0.1
langchain-anthropic>=0.1
langgraph>=0.2
langsmith>=0.1

# RAG
chromadb>=0.4
faiss-cpu>=1.8
qdrant-client>=1.8
pinecone-client>=3.0
llama-index>=0.10

# Agent Frameworks
crewai>=0.28
dspy-ai>=2.4

# Evaluation
ragas>=0.1
deepeval>=0.21

# Serving
fastapi>=0.110
uvicorn>=0.28
vllm>=0.4

# Utilities
python-dotenv>=1.0
tqdm>=4.66
rich>=13.7
httpx>=0.27
pydantic>=2.6
```

- [ ] **Step 3: Create notebook template**

Create a template notebook at `notebooks/template.ipynb` that establishes the standard structure. This is a JSON file. The template should have these cells:

Cell 1 (markdown):
```markdown
# [Topic Title]
> Section XX — Topic YY — [foundations|advanced|expert|build]

**Prerequisites:** [list prior notebooks]

**What you'll learn:**
- Point 1
- Point 2
- Point 3

**What you'll build:**
- Concrete deliverable
```

Cell 2 (markdown):
```markdown
## Setup
```

Cell 3 (code):
```python
# Dependencies
import torch
import numpy as np

# Verify setup
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

Cell 4 (markdown):
```markdown
## 1. [First Concept]

[Theory explanation with math where relevant]
```

Cell 5 (code):
```python
# Implementation
```

(Pattern repeats: markdown theory → code implementation → markdown theory → code...)

Final cell (markdown):
```markdown
## Summary

**Key takeaways:**
- Takeaway 1
- Takeaway 2

**Next:** [Link to next notebook]
```

- [ ] **Step 4: Create directory structure for all 22 sections**

```bash
# Section 01
mkdir -p 01-llms-deep-internals/{01-tokenization,02-embeddings,03-attention-mechanisms,04-transformer-architectures,05-pretraining,06-post-training,07-inference-engine,08-reasoning-models,capstone/build-your-own-llm}

# Section 02
mkdir -p 02-hugging-face-ecosystem/{01-transformers-library,02-datasets-and-data,03-peft-and-trl,04-accelerate-and-training,05-inference-and-deployment,06-spaces-and-demos,capstone/hf-model-pipeline}

# Section 03
mkdir -p 03-foundation-models/{01-model-families,02-model-selection,03-context-engineering,04-structured-output,05-prompting-mastery,06-local-inference,07-universal-apis,08-confidence-and-uncertainty,capstone/model-evaluation-platform}

# Section 04
mkdir -p 04-rag/{01-embeddings-and-indexing,02-chunking,03-vector-stores,04-retrieval,05-query-processing,06-advanced-rag-architectures,07-graph-rag,08-multimodal-rag,09-rag-evaluation,10-llamaindex,11-open-source-rag-stack,12-production-rag-operations,capstone/production-rag-system}

# Section 05
mkdir -p 05-langchain-ecosystem/{01-langchain-core,02-langchain-components,03-langchain-tools,04-langchain-retrieval,05-langgraph-fundamentals,06-langgraph-control-flow,07-langgraph-persistence,08-langgraph-human-in-the-loop,09-langgraph-streaming,10-langgraph-subgraphs,11-langgraph-memory,12-langgraph-advanced-patterns,13-langsmith,14-langgraph-platform,15-open-source-observability,capstone/full-stack-langgraph}

# Section 06
mkdir -p 06-agent-frameworks/{01-langgraph,02-openai-agents-sdk,03-claude-agent-sdk,04-crewai,05-autogen,06-dspy,07-semantic-kernel,08-haystack,09-llamaindex-agents,10-mcp,11-framework-selection,capstone/same-agent-three-ways}

# Section 07
mkdir -p 07-agent-architectures/{01-single-agent-patterns,02-planning-agents,03-multi-agent-systems,04-memory-architectures,05-coding-agents,06-research-and-search-agents,07-text-to-sql-agents,08-agent-evaluation,09-agent-safety,capstone/autonomous-agent-system}

# Section 08
mkdir -p 08-vision/{01-convnets,02-vision-transformers,03-object-detection,04-segmentation,05-image-generation,06-3d-vision,capstone/vision-ai-platform}

# Section 09
mkdir -p 09-audio/{01-audio-fundamentals,02-speech-recognition,03-speaker-analysis,04-text-to-speech,05-music-and-audio-generation,capstone/voice-ai-assistant}

# Section 10
mkdir -p 10-video/{01-video-fundamentals,02-video-understanding,03-tracking,04-video-generation,capstone/video-intelligence}

# Section 11
mkdir -p 11-multimodal/{01-contrastive-models,02-vision-language-models,03-document-ai,04-any-to-any-models,capstone/multimodal-ai-assistant}

# Section 12
mkdir -p 12-fine-tuning/{01-data-engineering,02-parameter-efficient,03-full-fine-tuning,04-vision-and-multimodal-ft,05-evaluation,06-open-source-ft-tools,07-data-curation,08-synthetic-data,09-model-distillation,capstone/domain-expert-model}

# Section 13
mkdir -p 13-evaluation/{01-llm-evaluation,02-rag-evaluation,03-agent-evaluation,04-safety-evaluation,05-production-evaluation,06-open-source-eval,capstone/eval-platform}

# Section 14
mkdir -p 14-safety-and-alignment/{01-prompt-security,02-guardrails,03-alignment,04-responsible-ai,05-auth-and-compliance,capstone/safe-ai-system}

# Section 15
mkdir -p 15-ai-system-design/{01-design-patterns,02-cost-engineering,03-caching-patterns,04-latency-optimization,05-error-handling-and-resilience,06-system-architectures,07-cloud-platforms,08-ui-and-ux,09-case-studies,capstone/system-design-portfolio}

# Section 16
mkdir -p 16-data-engineering-for-ai/{01-document-parsing,02-data-pipelines,03-web-data,capstone/data-platform}

# Section 17
mkdir -p 17-testing-and-ci-for-ai/{01-unit-testing,02-integration-testing,03-ci-cd-for-ai,04-load-and-performance,capstone/tested-ai-system}

# Section 18
mkdir -p 18-workflow-orchestration/{01-temporal,02-event-driven-ai,03-pipeline-orchestration,capstone/orchestrated-ai-platform}

# Section 19
mkdir -p 19-real-time-ai/{01-streaming-fundamentals,02-voice-pipelines,03-real-time-vision,04-live-multimodal,capstone/real-time-ai-assistant}

# Section 20
mkdir -p 20-edge-and-on-device/{01-model-compression,02-on-device-inference,03-hybrid-architectures,capstone/edge-ai-platform}

# Section 21
mkdir -p 21-deployment-and-mlops/{01-serving,02-optimization,03-infrastructure,04-mlops,05-monitoring,capstone/production-ai-platform}

# Section 22
mkdir -p 22-research-papers
```

- [ ] **Step 5: Commit scaffolding**

```bash
git add README.md requirements.txt notebooks/template.ipynb .gitignore
# Add all empty section directories with .gitkeep files
find . -type d -empty -not -path './.git/*' -exec touch {}/.gitkeep \;
git add .
git commit -m "scaffold: repo structure, README, requirements, notebook template"
```

---

## Phase 1: Section 01 — LLMs Deep Internals

This section is built in full detail to establish the pattern for all subsequent sections. Each topic gets its own task.

### Task 1.1: 01-tokenization/foundations.ipynb

**Files:**
- Create: `01-llms-deep-internals/01-tokenization/foundations.ipynb`

**Notebook content outline — this sets the template for ALL foundations notebooks:**

- [ ] **Step 1: Create the notebook**

The notebook should cover these sections (each section = markdown explanation + code implementation):

```
# Tokenization — Foundations
> Section 01 — Topic 01 — foundations

Prerequisites: None (first notebook in the curriculum)

What you'll learn:
- Why tokenization matters and how it affects model behavior
- How BPE, WordPiece, and Unigram tokenizers work internally
- How SentencePiece unifies these approaches

What you'll build:
- A BPE tokenizer from scratch in pure Python

## Setup
- pip install tokenizers transformers
- import dependencies

## 1. Why Tokenization Matters
- The bridge between text and numbers
- How tokenization choices affect: vocabulary size, out-of-vocab handling,
  multilingual performance, downstream task quality
- Show example: same text tokenized differently by different models

## 2. Character-Level vs Subword vs Word-Level
- Tradeoffs of each approach
- Why subword tokenization won (vocabulary efficiency + handling unseen words)
- Code: implement naive character-level and word-level tokenizers, show limitations

## 3. Byte Pair Encoding (BPE)
- Algorithm walkthrough with concrete example
- Code: implement BPE training from scratch
  - Start with character vocabulary
  - Count pair frequencies
  - Merge most frequent pair
  - Repeat until vocabulary size reached
- Code: implement BPE encoding (apply merges to new text)
- Code: implement BPE decoding
- Visualize: show merge history, vocabulary growth curve

## 4. WordPiece
- How it differs from BPE (likelihood-based vs frequency-based merging)
- Code: implement WordPiece training from scratch
- Code: implement WordPiece encoding with ## prefix convention
- Compare: same corpus tokenized by BPE vs WordPiece

## 5. Unigram Language Model
- Probabilistic approach: start with large vocabulary, prune
- Code: implement Unigram tokenizer training
  - Initialize large vocab from substrings
  - Compute token probabilities
  - Remove tokens with lowest loss impact
  - Repeat until target vocab size
- Code: implement Viterbi-based encoding (find most likely segmentation)

## 6. SentencePiece
- How it wraps BPE and Unigram with raw text processing (no pre-tokenization)
- Code: train a SentencePiece model using the sentencepiece library
- Compare output with our from-scratch implementations

## 7. Using Hugging Face Tokenizers
- The `tokenizers` library (fast Rust implementations)
- Code: load and use GPT-2, BERT, Llama tokenizers
- Code: compare how each tokenizes the same text
- Visualize: token boundaries, special tokens, vocabulary overlap

## Summary
- Key takeaways
- Next: 01-tokenization/advanced.ipynb
```

- [ ] **Step 2: Run the notebook end-to-end to verify all cells execute**

```bash
cd 01-llms-deep-internals/01-tokenization
jupyter nbconvert --to notebook --execute foundations.ipynb --output foundations_executed.ipynb
# Verify no errors, then remove the executed copy
rm foundations_executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/01-tokenization/foundations.ipynb
git commit -m "feat(01): add tokenization foundations notebook"
```

### Task 1.2: 01-tokenization/advanced.ipynb

**Files:**
- Create: `01-llms-deep-internals/01-tokenization/advanced.ipynb`

- [ ] **Step 1: Create the notebook**

```
# Tokenization — Advanced
> Section 01 — Topic 01 — advanced

Prerequisites: 01-tokenization/foundations.ipynb

What you'll learn:
- How to train custom tokenizers for specific domains
- Multilingual tokenization challenges and solutions
- Byte-level BPE and how modern models handle it

What you'll build:
- A domain-specific tokenizer trained on a custom corpus

## Setup

## 1. Training Custom Tokenizers with HF tokenizers Library
- Full training pipeline: normalizer → pre-tokenizer → model → post-processor
- Code: configure and train BPE tokenizer from scratch using `tokenizers` library
- Code: configure and train WordPiece tokenizer
- Code: configure and train Unigram tokenizer
- Compare training speed, vocabulary quality

## 2. Domain-Specific Tokenization
- Why general-purpose tokenizers waste tokens on domain text (code, medical, legal)
- Code: tokenize code snippets with GPT-2 tokenizer, count wasted tokens
- Code: train a code-specific tokenizer on a Python corpus
- Compare: token counts, compression ratios, fertility scores

## 3. Multilingual Tokenization
- The "tokenization tax" for non-English languages
- Code: compare token counts for same-meaning text across 5 languages using Llama tokenizer
- Code: train a multilingual tokenizer with balanced language representation
- Analysis: vocabulary allocation vs language representation

## 4. Byte-Level BPE
- How GPT-2/GPT-4 use byte-level BPE (no unknown tokens ever)
- Code: implement byte-level BPE from scratch
- Compare: byte-level vs character-level vocabulary efficiency
- Special token handling

## 5. Tokenizer Configuration and Special Tokens
- Chat templates, BOS/EOS/PAD tokens, system/user/assistant markers
- Code: examine chat templates from Llama 3, ChatGPT, Claude
- Code: build a custom chat template for a trained tokenizer

## 6. Tokenizer Evaluation Metrics
- Fertility (tokens per word), compression ratio, unknown token rate
- Code: build evaluation function, compare tokenizers on standard benchmarks

## Summary
- Next: 01-tokenization/expert.ipynb
```

- [ ] **Step 2: Run notebook, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/01-tokenization/advanced.ipynb
git commit -m "feat(01): add tokenization advanced notebook"
```

### Task 1.3: 01-tokenization/expert.ipynb

**Files:**
- Create: `01-llms-deep-internals/01-tokenization/expert.ipynb`

- [ ] **Step 1: Create the notebook**

```
# Tokenization — Expert
> Section 01 — Topic 01 — expert

Prerequisites: 01-tokenization/advanced.ipynb

What you'll learn:
- How tokenizer choices interact with model architecture and training
- Fertility analysis and its impact on inference cost
- Tokenizer merging and extension for continual pretraining

What you'll build:
- A tokenizer analysis toolkit that audits any model's tokenizer

## Setup

## 1. Tokenizer-Model Interactions
- How vocabulary size affects embedding matrix size (and memory)
- Relationship between token fertility and inference cost (more tokens = more FLOPs)
- Code: calculate FLOPs difference between tokenizers on same text
- How tokenization boundaries affect model's ability to learn word semantics

## 2. Fertility Analysis Deep Dive
- Definition: average tokens per word across a corpus
- Code: build fertility analyzer across languages and domains
- Visualize: fertility heatmaps by language and domain for popular models
- The economic argument: how fertility directly maps to API costs

## 3. Compression Ratios and Information Theory
- Relationship between tokenizer and entropy
- Code: measure bits-per-character for different tokenizers
- Compare: how close are learned tokenizers to optimal compression?

## 4. Tokenizer Extension and Merging
- Adding tokens to a pretrained tokenizer (for domain adaptation)
- Code: extend Llama tokenizer with domain vocabulary
- Initialize new token embeddings (mean, random, semantic)
- Code: merge two tokenizers (for combining domain expertise)
- Pitfalls: breaking existing model behavior

## 5. Token Healing
- The problem: tokenizer artifacts at generation boundaries
- Code: implement token healing for constrained generation
- When and why this matters for structured output

## 6. Tokenizer-Aware Prompt Optimization
- How to minimize token count while preserving meaning
- Code: build prompt compressor that respects token boundaries
- Practical savings calculation for production workloads

## Summary
- Next: 01-tokenization/build.ipynb
```

- [ ] **Step 2: Run notebook, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/01-tokenization/expert.ipynb
git commit -m "feat(01): add tokenization expert notebook"
```

### Task 1.4: 01-tokenization/build.ipynb

**Files:**
- Create: `01-llms-deep-internals/01-tokenization/build.ipynb`

- [ ] **Step 1: Create the notebook**

```
# Tokenization — Build: Train & Evaluate a Domain Tokenizer
> Section 01 — Topic 01 — build

Prerequisites: All tokenization notebooks

What you'll build:
- A complete domain-specific tokenizer: trained, evaluated, and packaged
- Corpus: Python source code from a real open-source project
- Full evaluation against general-purpose tokenizers

## Setup
- Download corpus (e.g., CPython source code via datasets library)

## 1. Corpus Preparation
- Code: download and preprocess Python source code
- Code: split into train/eval sets
- Code: analyze corpus statistics (vocabulary, character distribution)

## 2. Train Three Tokenizers
- Code: train BPE tokenizer on the corpus (vocab_size=32000)
- Code: train Unigram tokenizer on the corpus (vocab_size=32000)
- Code: train byte-level BPE tokenizer on the corpus (vocab_size=32000)

## 3. Comparative Evaluation
- Code: measure fertility, compression ratio, unknown token rate on eval set
- Code: compare against GPT-2, Llama 3, CodeLlama tokenizers on same eval set
- Code: measure tokenization speed (tokens/second)
- Visualize: comparison table and charts

## 4. Qualitative Analysis
- Code: show side-by-side tokenization of representative code snippets
- Analyze: which tokenizer best preserves code structure?
- Analyze: which handles edge cases (unicode, long identifiers, docstrings)?

## 5. Package and Export
- Code: save best tokenizer in HF format
- Code: create tokenizer card (metadata, training details, benchmarks)
- Code: load and verify the saved tokenizer

## Summary
- What we learned about domain-specific tokenization
- Next: 02-embeddings/foundations.ipynb
```

- [ ] **Step 2: Run notebook, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/01-tokenization/build.ipynb
git commit -m "feat(01): add tokenization build notebook"
```

### Task 1.5: 02-embeddings (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/02-embeddings/foundations.ipynb`
- Create: `01-llms-deep-internals/02-embeddings/advanced.ipynb`
- Create: `01-llms-deep-internals/02-embeddings/expert.ipynb`
- Create: `01-llms-deep-internals/02-embeddings/build.ipynb`

- [ ] **Step 1: Create foundations.ipynb**

```
# Embeddings — Foundations

## 1. Why Embeddings? From One-Hot to Dense
- One-hot encoding limitations, the curse of dimensionality
- Code: implement one-hot encoding, show sparsity problem

## 2. Word2Vec — Skip-gram
- Architecture, training objective (predict context from center word)
- Code: implement skip-gram from scratch in PyTorch
- Train on a small corpus, visualize learned embeddings with t-SNE

## 3. Word2Vec — CBOW
- Architecture, training objective (predict center from context)
- Code: implement CBOW from scratch
- Compare: skip-gram vs CBOW on same corpus

## 4. GloVe
- Co-occurrence matrix approach, weighted least squares objective
- Code: build co-occurrence matrix from corpus
- Code: implement GloVe training from scratch
- Compare: GloVe vs Word2Vec embeddings

## 5. Evaluating Embeddings
- Analogy tasks (king - man + woman = queen)
- Code: implement analogy solver
- Similarity benchmarks, clustering visualization

## Summary → Next: advanced.ipynb
```

- [ ] **Step 2: Create advanced.ipynb**

```
# Embeddings — Advanced

## 1. From Static to Contextual Embeddings
- Limitation of static embeddings (one vector per word regardless of context)
- ELMo: bidirectional LSTM contextualization
- Code: extract contextual embeddings from ELMo, show same word in different contexts

## 2. Sentence Embeddings
- sentence-transformers library, bi-encoder architecture
- Code: generate sentence embeddings, compute semantic similarity
- Mean pooling vs CLS token vs attention-weighted pooling

## 3. Modern Embedding Models
- Survey: OpenAI text-embedding-3, Cohere Embed, BGE, E5, Jina, Nomic
- Code: compare embeddings from 5+ models on standard tasks
- MTEB leaderboard: what the benchmarks actually measure

## 4. Choosing Embedding Models
- Task-specific selection: retrieval, classification, clustering, STS
- Dimension tradeoffs: larger = more expressive but more expensive
- Code: benchmark models on domain-specific data

## 5. Embedding Similarity Metrics
- Cosine, dot product, Euclidean: when to use which
- Code: implement all three, show when they diverge
- How normalization affects metric choice

## Summary → Next: expert.ipynb
```

- [ ] **Step 3: Create expert.ipynb**

```
# Embeddings — Expert

## 1. Embedding Space Geometry
- Isotropy vs anisotropy in embedding spaces
- Code: measure isotropy of different model embeddings
- Why anisotropic spaces hurt retrieval and how to fix it

## 2. Probing Embeddings
- What information is encoded in embeddings?
- Code: train linear probes for POS, NER, sentiment on frozen embeddings
- Compare: what different models encode at different layers

## 3. Matryoshka Embeddings
- Train once, use at any dimension (truncate without retraining)
- Code: implement Matryoshka loss
- Benchmark: quality vs dimension for different truncation points

## 4. Binary and Scalar Quantization
- Reducing embedding storage: float32 → int8 → binary
- Code: implement scalar quantization and binary quantization
- Benchmark: recall degradation vs storage/speed improvement

## 5. Fine-Tuning Embeddings for Domain-Specific Retrieval
- Contrastive loss, triplet loss, multiple negatives ranking loss
- Code: fine-tune a sentence-transformer on domain-specific pairs
- Evaluate: before vs after on domain retrieval benchmark

## Summary → Next: build.ipynb
```

- [ ] **Step 4: Create build.ipynb**

```
# Embeddings — Build: Fine-Tuned Embedding Model + MTEB Evaluation

## 1. Define Domain and Collect Data
- Choose a domain (e.g., technical documentation)
- Code: create training pairs from real data

## 2. Fine-Tune Embedding Model
- Code: fine-tune sentence-transformer with MNRL loss
- Monitor training with evaluation callback

## 3. Evaluate on MTEB Tasks
- Code: run MTEB evaluation on standard tasks
- Compare: base model vs fine-tuned on domain tasks

## 4. Quantization and Deployment
- Code: quantize to int8, benchmark quality retention
- Code: export model, create model card

## Summary → Next: 03-attention-mechanisms/foundations.ipynb
```

- [ ] **Step 5: Run all notebooks, verify execution**
- [ ] **Step 6: Commit**

```bash
git add 01-llms-deep-internals/02-embeddings/
git commit -m "feat(01): add embeddings topic (foundations, advanced, expert, build)"
```

### Task 1.6: 03-attention-mechanisms (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/03-attention-mechanisms/{foundations,advanced,expert,build}.ipynb`

- [ ] **Step 1: Create all 4 notebooks**

**foundations.ipynb:**
```
## 1. Attention Intuition — from seq2seq bottleneck to attending to everything
## 2. Scaled Dot-Product Attention — implement from scratch, step by step
## 3. Multi-Head Attention — implement, show how heads specialize
## 4. Self-Attention vs Cross-Attention — when each is used, implement both
## 5. Positional Encoding — sinusoidal and learned, implement both
## 6. Masking — padding masks, causal masks, implement and visualize
```

**advanced.ipynb:**
```
## 1. Multi-Query Attention (MQA) — shared KV heads, implement from scratch
## 2. Grouped-Query Attention (GQA) — the middle ground, implement
## 3. Flash Attention — the memory-efficient algorithm, explain IO complexity
## 4. Ring Attention — distributed attention across devices
## 5. Sliding Window Attention — Mistral's approach, implement
## 6. Benchmarking — compare all attention variants on speed and memory
```

**expert.ipynb:**
```
## 1. Attention Sinks — why first tokens get disproportionate attention
## 2. Attention Pattern Analysis — extract and visualize attention from real models
## 3. Sparse Attention — BigBird, Longformer patterns, implement local+global
## 4. Linear Attention — kernel-based approximations, implement
## 5. Attention in Practice — how production models combine these techniques
```

**build.ipynb:**
```
## Build: Implement Flash Attention and Benchmark
- Implement naive attention with O(n^2) memory
- Implement tiled/flash attention with O(n) memory
- Benchmark both on increasing sequence lengths
- Plot: memory usage, throughput, wall-clock time
- Compare against torch.nn.functional.scaled_dot_product_attention
```

- [ ] **Step 2: Run all notebooks, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/03-attention-mechanisms/
git commit -m "feat(01): add attention mechanisms topic"
```

### Task 1.7: 04-transformer-architectures (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/04-transformer-architectures/{foundations,advanced,expert,build}.ipynb`

- [ ] **Step 1: Create all 4 notebooks**

**foundations.ipynb:**
```
## 1. The Original Transformer — encoder-decoder, implement from "Attention Is All You Need"
## 2. Encoder-Only (BERT-style) — implement, explain MLM objective
## 3. Decoder-Only (GPT-style) — implement, explain autoregressive LM objective
## 4. Feed-Forward Networks — role, implement with ReLU and GELU
## 5. Layer Normalization — pre-norm vs post-norm, implement both
## 6. Residual Connections — why they matter, gradient flow analysis
## 7. Putting It All Together — assemble complete decoder-only transformer
```

**advanced.ipynb:**
```
## 1. Rotary Position Embeddings (RoPE) — math, implement from scratch
## 2. ALiBi (Attention with Linear Biases) — implement, compare to RoPE
## 3. RMSNorm — why models switched from LayerNorm, implement
## 4. SwiGLU Activation — implement, benchmark against GELU
## 5. Parallel Attention + FFN — PaLM-style parallel layers, implement
## 6. Modern Architecture Recipe — Llama 3 style: RoPE + RMSNorm + SwiGLU + GQA
```

**expert.ipynb:**
```
## 1. Mixture of Experts (MoE) — routing, load balancing, implement
## 2. Switch Transformer — simplified MoE routing, implement
## 3. Expert Choice Routing — capacity factor, implement and compare
## 4. Sparse vs Dense Tradeoffs — when MoE wins, when it doesn't
## 5. Architecture Scaling — how to decide: depth, width, heads, FFN ratio
```

**build.ipynb:**
```
## Build: GPT From Scratch (nanoGPT Extended)
- Implement complete GPT-2 architecture from scratch
- Add modern improvements: RoPE, RMSNorm, SwiGLU, GQA
- Train on a small text corpus (e.g., Shakespeare or WikiText)
- Generate text, analyze attention patterns
- Compare: vanilla vs modern architecture on same training budget
```

- [ ] **Step 2: Run all notebooks, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/04-transformer-architectures/
git commit -m "feat(01): add transformer architectures topic"
```

### Task 1.8: 05-pretraining (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/05-pretraining/{foundations,advanced,expert,build}.ipynb`

- [ ] **Step 1: Create all 4 notebooks**

**foundations.ipynb:**
```
## 1. Language Modeling Objectives — CLM, MLM, prefix LM, explain and compare
## 2. Data Pipelines — tokenized datasets, packing, padding strategies
## 3. Training Loop — implement basic pretraining loop with PyTorch
## 4. Learning Rate Schedules — warmup, cosine decay, implement and visualize
## 5. Loss Curves — what to watch for, diagnosing training problems
## 6. Checkpointing — saving/loading model state, resuming training
```

**advanced.ipynb:**
```
## 1. Distributed Training Fundamentals — DDP, why single GPU isn't enough
## 2. DeepSpeed ZeRO — stages 1, 2, 3 explained, code: setup DeepSpeed config
## 3. FSDP (Fully Sharded Data Parallel) — PyTorch native, code: configure FSDP
## 4. Mixed Precision Training — fp16, bf16, fp8, loss scaling, implement
## 5. Gradient Accumulation — simulating larger batches, implement
## 6. Data Loading at Scale — streaming datasets, data parallelism across workers
```

**expert.ipynb:**
```
## 1. Scaling Laws — Kaplan et al., Chinchilla: derive compute-optimal model size
## 2. Compute-Optimal Training — given a GPU budget, what model size and token count?
## 3. Data Mixtures — how to balance domains (web, books, code, math)
## 4. Data Quality — deduplication (MinHash), filtering (perplexity-based), toxicity removal
## 5. Training Dynamics — loss spikes, instabilities, gradient norms, debugging
## 6. Emergent Abilities — what appears at scale, phase transitions, controversy
```

**build.ipynb:**
```
## Build: Pretrain a Small Language Model
- Define model: ~125M params with modern architecture (from Task 1.7 build)
- Prepare dataset: 1-5B tokens from C4/RedPajama subset
- Configure training: DeepSpeed ZeRO-2, bf16, cosine schedule
- Train for compute-optimal token count per Chinchilla
- Evaluate: perplexity on held-out set, compare to published baselines
- Verify scaling law predictions match observed loss
```

- [ ] **Step 2: Run all notebooks, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/05-pretraining/
git commit -m "feat(01): add pretraining topic"
```

### Task 1.9: 06-post-training (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/06-post-training/{foundations,advanced,expert,build}.ipynb`

- [ ] **Step 1: Create all 4 notebooks**

**foundations.ipynb:**
```
## 1. Why Post-Training? — base models vs instruction-tuned vs chat
## 2. Supervised Fine-Tuning (SFT) — instruction dataset format, implement training loop
## 3. Chat Templates — how models learn turn structure, implement template application
## 4. Instruction Dataset Design — what makes a good instruction dataset
## 5. SFT in Practice — code: fine-tune a small model on Alpaca-style data with TRL
```

**advanced.ipynb:**
```
## 1. Reward Modeling — what is a reward model, implement training
## 2. RLHF with PPO — the original InstructGPT approach, implement key components
## 3. DPO (Direct Preference Optimization) — math, why it replaces PPO, implement loss
## 4. DPO in Practice — code: train DPO with TRL on preference data
## 5. Comparing RLHF vs DPO — when to use which, empirical tradeoffs
```

**expert.ipynb:**
```
## 1. Constitutional AI — principle-based training without human labels, implement
## 2. ORPO — Odds Ratio Preference Optimization, simpler than DPO, implement
## 3. KTO — Kahneman-Tversky Optimization, works without pairs, implement
## 4. Process Reward Models — step-level rewards vs outcome rewards
## 5. Iterative RLHF — multiple rounds of alignment, handling reward hacking
## 6. Rejection Sampling — generate multiple, pick best, use as training data
```

**build.ipynb:**
```
## Build: Full Post-Training Pipeline
- Start with pretrained model (from Task 1.8 or use a small open model)
- Phase 1: SFT on curated instruction dataset
- Phase 2: DPO on preference pairs
- Evaluate at each stage: MT-Bench style eval, compare base vs SFT vs DPO
- Analyze: where does each stage help most?
```

- [ ] **Step 2: Run all notebooks, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/06-post-training/
git commit -m "feat(01): add post-training topic"
```

### Task 1.10: 07-inference-engine (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/07-inference-engine/{foundations,advanced,expert,build}.ipynb`

- [ ] **Step 1: Create all 4 notebooks**

**foundations.ipynb:**
```
## 1. Autoregressive Generation — the decode loop, implement from scratch
## 2. Decoding Strategies — greedy, beam search, implement each
## 3. Sampling — temperature, top-k, top-p (nucleus), implement each
## 4. KV Caching — why it matters, implement cached generation, benchmark speedup
## 5. Stopping Criteria — EOS token, max length, custom stopping
## 6. Structured Output — constrained decoding basics, JSON mode
```

**advanced.ipynb:**
```
## 1. Speculative Decoding — draft model + verification, implement from scratch
## 2. Continuous Batching — dynamic batching for throughput, explain vLLM's approach
## 3. Paged Attention — virtual memory for KV cache, explain PagedAttention
## 4. Beam Search Variants — diverse beam search, group beam search
## 5. Repetition Penalties — frequency and presence penalties, implement
## 6. Logit Processors — implementing custom logit manipulation pipeline
```

**expert.ipynb:**
```
## 1. Quantization Deep Dive — GPTQ: per-column quantization, implement
## 2. AWQ — activation-aware quantization, how it preserves salient weights
## 3. GGUF — llama.cpp format, mixed quantization, k-quant schemes
## 4. fp8 and int4 — emerging precision formats, hardware support
## 5. Kernel Optimization — how vLLM/TGI optimize GPU utilization
## 6. Benchmarking Inference — tokens/sec, time-to-first-token, throughput vs latency
```

**build.ipynb:**
```
## Build: Deploy with vLLM and Benchmark
- Quantize a model: GPTQ, AWQ, and GGUF variants
- Deploy each with vLLM
- Benchmark: throughput, latency, memory, quality (perplexity)
- Build comparison dashboard showing tradeoffs
- Determine optimal configuration for different use cases
```

- [ ] **Step 2: Run all notebooks, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/07-inference-engine/
git commit -m "feat(01): add inference engine topic"
```

### Task 1.11: 08-reasoning-models (all 4 notebooks)

**Files:**
- Create: `01-llms-deep-internals/08-reasoning-models/{foundations,advanced,expert,build}.ipynb`

- [ ] **Step 1: Create all 4 notebooks**

**foundations.ipynb:**
```
## 1. Chain-of-Thought Prompting — why it works, implement and evaluate
## 2. Inference-Time Compute Scaling — more thinking = better answers, the research
## 3. Self-Consistency — sample multiple chains, majority vote, implement
## 4. Step-by-Step Reasoning — decomposition strategies, implement
## 5. Comparing Reasoning Approaches — CoT vs self-consistency vs step-by-step on GSM8K
```

**advanced.ipynb:**
```
## 1. o1/o3-Style Reasoning — thinking tokens, extended internal monologue
## 2. Reasoning Traces — what happens inside reasoning models
## 3. Structured Reasoning — forcing models into explicit reasoning frameworks
## 4. Tool-Augmented Reasoning — using code execution for verification
## 5. Reasoning with Claude — extended thinking API, practical patterns
```

**expert.ipynb:**
```
## 1. Process Reward Models — training step-level reward models
## 2. MCTS for Reasoning — Monte Carlo Tree Search applied to language models
## 3. Verification and Self-Correction — how models check their own work
## 4. Training Reasoning Models — approaches to improving reasoning during training
## 5. Limits of Reasoning — what current models can and can't reason about
```

**build.ipynb:**
```
## Build: Reasoning Pipeline with Verification
- Build a math reasoning pipeline that:
  1. Generates multiple reasoning chains
  2. Verifies each step with code execution
  3. Scores chains using a reward model
  4. Selects the best answer with confidence
- Evaluate on GSM8K and MATH subsets
- Compare: pipeline accuracy vs single-shot prompting
```

- [ ] **Step 2: Run all notebooks, verify execution**
- [ ] **Step 3: Commit**

```bash
git add 01-llms-deep-internals/08-reasoning-models/
git commit -m "feat(01): add reasoning models topic"
```

### Task 1.12: Section 01 Capstone

**Files:**
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/README.md`
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/01-pretrain.ipynb`
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/02-sft.ipynb`
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/03-dpo.ipynb`
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/04-quantize.ipynb`
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/05-serve.ipynb`
- Create: `01-llms-deep-internals/capstone/build-your-own-llm/06-evaluate.ipynb`

- [ ] **Step 1: Create README.md**

```markdown
# Capstone: Build Your Own LLM

End-to-end project: pretrain → SFT → DPO → quantize → serve → evaluate.

This capstone ties together everything from Section 01. You will:

1. **Pretrain** a ~125M parameter language model with modern architecture
2. **SFT** on curated instruction data
3. **DPO** on preference pairs
4. **Quantize** to int4 for efficient serving
5. **Serve** via vLLM with a FastAPI wrapper
6. **Evaluate** at each stage with automated benchmarks

## Prerequisites
- All Section 01 notebooks completed
- GPU with 16GB+ VRAM (or Colab/cloud GPU)

## Notebooks
Run in order: 01 through 06.
```

- [ ] **Step 2: Create each capstone notebook**

Each notebook builds on the previous one's output:

**01-pretrain.ipynb:** Define model architecture (RoPE + RMSNorm + SwiGLU + GQA, ~125M params). Prepare 1B tokens from RedPajama subset. Train with Accelerate + DeepSpeed. Save checkpoint.

**02-sft.ipynb:** Load pretrained checkpoint. Fine-tune on subset of OpenHermes/Alpaca. Use TRL SFTTrainer. Save SFT checkpoint.

**03-dpo.ipynb:** Load SFT checkpoint. Train DPO on UltraFeedback subset. Compare before/after on sample prompts.

**04-quantize.ipynb:** Quantize to GPTQ int4 and GGUF Q4_K_M. Benchmark: perplexity, model size, inference speed.

**05-serve.ipynb:** Deploy quantized model with vLLM. Build FastAPI wrapper with streaming. Load test with concurrent requests.

**06-evaluate.ipynb:** Run evaluation at each stage (base, SFT, DPO). Metrics: perplexity, MT-Bench style, task-specific. Build comparison report.

- [ ] **Step 3: Run all capstone notebooks in sequence**
- [ ] **Step 4: Commit**

```bash
git add 01-llms-deep-internals/capstone/
git commit -m "feat(01): add capstone — build your own LLM end-to-end"
```

### Task 1.13: Section 01 review and polish

- [ ] **Step 1: Verify all notebooks run end-to-end**

```bash
for nb in $(find 01-llms-deep-internals -name "*.ipynb" | sort); do
  echo "Testing: $nb"
  jupyter nbconvert --to notebook --execute "$nb" --ExecutePreprocessor.timeout=600 --output /dev/null 2>&1 && echo "  PASS" || echo "  FAIL"
done
```

- [ ] **Step 2: Verify cross-references between notebooks (Next links)**
- [ ] **Step 3: Final commit for Section 01**

```bash
git add -A
git commit -m "feat(01): complete Section 01 — LLMs Deep Internals"
```

---

## Phase 2 through Phase 22: Remaining Sections

Each subsequent section follows the exact same pattern established in Phase 1. Each gets its own detailed plan document when ready to build.

### Section 02: Hugging Face Ecosystem
**Plan:** `docs/superpowers/plans/section-02-hugging-face-ecosystem.md`
**Topics (6):** transformers-library, datasets-and-data, peft-and-trl, accelerate-and-training, inference-and-deployment, spaces-and-demos
**Capstone:** hf-model-pipeline (train → evaluate → publish → deploy → demo)
**Key dependencies:** Section 01 (uses pretrained models and concepts from there)
**Estimated notebooks:** 24 + capstone

### Section 03: Foundation Models
**Plan:** `docs/superpowers/plans/section-03-foundation-models.md`
**Topics (8):** model-families, model-selection, context-engineering, structured-output, prompting-mastery, local-inference, universal-apis, confidence-and-uncertainty
**Capstone:** model-evaluation-platform (benchmark + router + cost optimizer)
**Key dependencies:** Sections 01-02
**Estimated notebooks:** 32 + capstone
**Notes:** model-families topic uses individual .ipynb per family (6 notebooks) instead of foundations/advanced/expert/build pattern

### Section 04: RAG
**Plan:** `docs/superpowers/plans/section-04-rag.md`
**Topics (12):** embeddings-and-indexing, chunking, vector-stores, retrieval, query-processing, advanced-rag-architectures, graph-rag, multimodal-rag, rag-evaluation, llamaindex, open-source-rag-stack, production-rag-operations
**Capstone:** production-rag-system (multi-source + graph + multimodal + eval + monitoring)
**Key dependencies:** Sections 01-03 (especially embeddings, model APIs)
**Estimated notebooks:** 48 + capstone

### Section 05: LangChain Ecosystem
**Plan:** `docs/superpowers/plans/section-05-langchain-ecosystem.md`
**Topics (15):** langchain-core, langchain-components, langchain-tools, langchain-retrieval, langgraph-fundamentals, langgraph-control-flow, langgraph-persistence, langgraph-human-in-the-loop, langgraph-streaming, langgraph-subgraphs, langgraph-memory, langgraph-advanced-patterns, langsmith, langgraph-platform, open-source-observability
**Capstone:** full-stack-langgraph (multi-agent + HITL + streaming + persistence + eval + deployed)
**Key dependencies:** Sections 01-04
**Estimated notebooks:** 60 + capstone

### Section 06: Agent Frameworks
**Plan:** `docs/superpowers/plans/section-06-agent-frameworks.md`
**Topics (11):** langgraph (cross-ref to S05), openai-agents-sdk, claude-agent-sdk, crewai, autogen, dspy, semantic-kernel, haystack, llamaindex-agents, mcp, framework-selection
**Capstone:** same-agent-three-ways (identical agent in 3 frameworks + eval)
**Key dependencies:** Section 05 (LangGraph), Sections 01-03
**Estimated notebooks:** 37 + capstone
**Notes:** Tier 1 (OpenAI, Claude) = 4 notebooks each. Tier 2 = 3 notebooks each. MCP + selection = 4 each.

### Section 07: Agent Architectures
**Plan:** `docs/superpowers/plans/section-07-agent-architectures.md`
**Topics (9):** single-agent-patterns, planning-agents, multi-agent-systems, memory-architectures, coding-agents, research-and-search-agents, text-to-sql-agents, agent-evaluation, agent-safety
**Capstone:** autonomous-agent-system (planning + memory + safety + eval)
**Key dependencies:** Sections 05-06
**Estimated notebooks:** 36 + capstone

### Section 08: Vision
**Plan:** `docs/superpowers/plans/section-08-vision.md`
**Topics (6):** convnets, vision-transformers, object-detection, segmentation, image-generation, 3d-vision
**Capstone:** vision-ai-platform (detection + segmentation + generation + 3D)
**Key dependencies:** Section 01 (attention, transformers)
**Estimated notebooks:** 24 + capstone

### Section 09: Audio
**Plan:** `docs/superpowers/plans/section-09-audio.md`
**Topics (5):** audio-fundamentals, speech-recognition, speaker-analysis, text-to-speech, music-and-audio-generation
**Capstone:** voice-ai-assistant (ASR + LLM + TTS + speaker ID)
**Key dependencies:** Section 01, Section 02 (HF tools for fine-tuning Whisper etc.)
**Estimated notebooks:** 20 + capstone

### Section 10: Video
**Plan:** `docs/superpowers/plans/section-10-video.md`
**Topics (4):** video-fundamentals, video-understanding, tracking, video-generation
**Capstone:** video-intelligence (understanding + tracking + generation)
**Key dependencies:** Section 08 (vision foundations)
**Estimated notebooks:** 16 + capstone

### Section 11: Multimodal
**Plan:** `docs/superpowers/plans/section-11-multimodal.md`
**Topics (4):** contrastive-models, vision-language-models, document-ai, any-to-any-models
**Capstone:** multimodal-ai-assistant (text + image + audio + document)
**Key dependencies:** Sections 08-10
**Estimated notebooks:** 16 + capstone

### Section 12: Fine-Tuning
**Plan:** `docs/superpowers/plans/section-12-fine-tuning.md`
**Topics (9):** data-engineering, parameter-efficient, full-fine-tuning, vision-and-multimodal-ft, evaluation, open-source-ft-tools, data-curation, synthetic-data, model-distillation
**Capstone:** domain-expert-model (curate → train → evaluate → deploy → monitor)
**Key dependencies:** Sections 01-02 (training fundamentals, HF tools)
**Estimated notebooks:** 36 + capstone

### Section 13: Evaluation
**Plan:** `docs/superpowers/plans/section-13-evaluation.md`
**Topics (6):** llm-evaluation, rag-evaluation, agent-evaluation, safety-evaluation, production-evaluation, open-source-eval
**Capstone:** eval-platform (unified eval across all systems)
**Key dependencies:** Sections 04-07 (need RAG/agent systems to evaluate)
**Estimated notebooks:** 24 + capstone

### Section 14: Safety and Alignment
**Plan:** `docs/superpowers/plans/section-14-safety-and-alignment.md`
**Topics (5):** prompt-security, guardrails, alignment, responsible-ai, auth-and-compliance
**Capstone:** safe-ai-system (security + guardrails + alignment + compliance)
**Key dependencies:** Sections 01, 05-07
**Estimated notebooks:** 20 + capstone

### Section 15: AI System Design
**Plan:** `docs/superpowers/plans/section-15-ai-system-design.md`
**Topics (9):** design-patterns, cost-engineering, caching-patterns, latency-optimization, error-handling-and-resilience, system-architectures, cloud-platforms, ui-and-ux, case-studies
**Capstone:** system-design-portfolio (3 complete system designs with prototypes)
**Key dependencies:** All prior sections (this is the synthesis section)
**Estimated notebooks:** 36 + capstone
**Notes:** case-studies topic uses individual .ipynb per case (5 notebooks) instead of standard pattern

### Section 16: Data Engineering for AI
**Plan:** `docs/superpowers/plans/section-16-data-engineering.md`
**Topics (3):** document-parsing, data-pipelines, web-data
**Capstone:** data-platform (multi-source ingestion + quality + versioning)
**Key dependencies:** Section 04 (RAG ingestion context)
**Estimated notebooks:** 12 + capstone

### Section 17: Testing and CI for AI
**Plan:** `docs/superpowers/plans/section-17-testing-and-ci.md`
**Topics (4):** unit-testing, integration-testing, ci-cd-for-ai, load-and-performance
**Capstone:** tested-ai-system (fully tested + CI/CD + performance validated)
**Key dependencies:** Sections 04-07 (need real AI systems to test)
**Estimated notebooks:** 16 + capstone

### Section 18: Workflow Orchestration
**Plan:** `docs/superpowers/plans/section-18-workflow-orchestration.md`
**Topics (3):** temporal, event-driven-ai, pipeline-orchestration
**Capstone:** orchestrated-ai-platform (durable agents + event-driven + scheduled)
**Key dependencies:** Sections 05-07 (agent workflows)
**Estimated notebooks:** 12 + capstone

### Section 19: Real-Time AI
**Plan:** `docs/superpowers/plans/section-19-real-time-ai.md`
**Topics (4):** streaming-fundamentals, voice-pipelines, real-time-vision, live-multimodal
**Capstone:** real-time-ai-assistant (voice + vision + screen)
**Key dependencies:** Sections 08-09 (vision, audio)
**Estimated notebooks:** 16 + capstone

### Section 20: Edge and On-Device
**Plan:** `docs/superpowers/plans/section-20-edge-and-on-device.md`
**Topics (3):** model-compression, on-device-inference, hybrid-architectures
**Capstone:** edge-ai-platform (on-device + cloud fallback + privacy)
**Key dependencies:** Sections 01, 08 (inference optimization, vision models)
**Estimated notebooks:** 12 + capstone

### Section 21: Deployment and MLOps
**Plan:** `docs/superpowers/plans/section-21-deployment-and-mlops.md`
**Topics (5):** serving, optimization, infrastructure, mlops, monitoring
**Capstone:** production-ai-platform (serve + optimize + scale + monitor + CI/CD)
**Key dependencies:** All prior sections
**Estimated notebooks:** 20 + capstone

### Section 22: Research Papers
**Plan:** `docs/superpowers/plans/section-22-research-papers.md`
**Papers (20):** Attention Is All You Need, BERT, GPT, CLIP, DDPM, LoRA, Flash Attention, RAG, ReAct, Self-RAG, DPO, GraphRAG, Mamba, MoE, ViT, SWE-Agent, Toolformer, CRAG, RAPTOR, MemGPT
**Key dependencies:** Sections 01-11 (concepts referenced in papers)
**Estimated notebooks:** 20
**Notes:** Each paper is a single notebook, no foundations/advanced/expert/build split

---

## Summary

| Phase | Section | Topics | Notebooks (approx) |
|-------|---------|--------|-------------------|
| 0 | Scaffolding | — | 1 template |
| 1 | 01-LLMs Deep Internals | 8 | 32 + capstone (6) |
| 2 | 02-HF Ecosystem | 6 | 24 + capstone |
| 3 | 03-Foundation Models | 8 | 32 + capstone |
| 4 | 04-RAG | 12 | 48 + capstone |
| 5 | 05-LangChain Ecosystem | 15 | 60 + capstone |
| 6 | 06-Agent Frameworks | 11 | 37 + capstone |
| 7 | 07-Agent Architectures | 9 | 36 + capstone |
| 8 | 08-Vision | 6 | 24 + capstone |
| 9 | 09-Audio | 5 | 20 + capstone |
| 10 | 10-Video | 4 | 16 + capstone |
| 11 | 11-Multimodal | 4 | 16 + capstone |
| 12 | 12-Fine-Tuning | 9 | 36 + capstone |
| 13 | 13-Evaluation | 6 | 24 + capstone |
| 14 | 14-Safety & Alignment | 5 | 20 + capstone |
| 15 | 15-AI System Design | 9 | 36 + capstone |
| 16 | 16-Data Engineering | 3 | 12 + capstone |
| 17 | 17-Testing & CI | 4 | 16 + capstone |
| 18 | 18-Workflow Orchestration | 3 | 12 + capstone |
| 19 | 19-Real-Time AI | 4 | 16 + capstone |
| 20 | 20-Edge & On-Device | 3 | 12 + capstone |
| 21 | 21-Deployment & MLOps | 5 | 20 + capstone |
| 22 | 22-Research Papers | 20 | 20 |
| **Total** | **22 sections** | **~130** | **~555** |

## Execution Instructions

When starting a new session to build this curriculum:

1. **Read this plan** at `docs/superpowers/plans/2026-04-11-ai-engineering-master-plan.md`
2. **Read the design spec** at `docs/superpowers/specs/2026-04-11-ai-engineering-curriculum-design.md`
3. **Check what's already built** — `find . -name "*.ipynb" | wc -l` and check git log
4. **Pick up where you left off** — work through sections in order (01, 02, 03...)
5. **For Phase 1 (Section 01):** tasks are fully detailed above — execute them in order
6. **For Phases 2-22:** write the detailed section plan first (following Phase 1 pattern), then execute
7. **Each topic = 4 notebooks** (foundations, advanced, expert, build) unless noted otherwise
8. **Each section ends with a capstone** project
9. **Commit after each topic** (not each notebook — one commit per topic keeps history clean)
10. **Test notebooks execute** before committing — `jupyter nbconvert --execute`
