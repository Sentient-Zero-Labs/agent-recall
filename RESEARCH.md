# Memory Systems for AI Agents — Deep Research

**Prepared for:** Recall v0.2 Design  
**Date:** May 2026  
**Scope:** Academic papers (2023–2026), production systems, mathematical formulations

---

## Table of Contents

1. [Memory Types and Representations](#1-memory-types-and-representations)
2. [Retrieval Math — What Actually Works](#2-retrieval-math--what-actually-works)
3. [What Queries Agents Actually Make](#3-what-queries-agents-actually-make)
4. [What Data the System Should Accept](#4-what-data-the-system-should-accept)
5. [Memory Operations Beyond Store/Retrieve](#5-memory-operations-beyond-storeretrieve)
6. [Paper Summaries](#6-paper-summaries)
7. [Design Implications for Recall v0.2](#7-design-implications-for-recall-v02)
8. [Resources](#8-resources)

---

## 1. Memory Types and Representations

### The Cognitive Science Foundation (Tulving Applied to LLMs)

The most cited framework in agent memory research is Tulving's taxonomy from cognitive psychology, formalized for LLMs in the **CoALA paper** (Sumers et al., 2023, arXiv:2309.02427) and extended in several 2024–2026 surveys. The four types matter because each answers a different class of query and requires different storage and retrieval mechanics.

**Working Memory**  
Temporary information held active for the current reasoning cycle. In LLM terms, this is the active context window. CoALA defines it as "a data structure that persists across LLM calls, where on each call the LLM input is synthesized from a subset of working memory." It is implicitly managed by context management systems like MemGPT — not by a persistent store. Recall does not need to manage this; the calling application does. However, what Recall *retrieves* gets inserted into working memory, so retrieval precision directly controls context quality.

**Episodic Memory**  
Specific experiences with temporal context: what happened, when, in what sequence. The **"Episodic Memory is the Missing Piece" paper** (Pink et al., 2025, arXiv:2502.06975) defines five required properties: (1) long-term storage beyond the session, (2) explicit reasoning over stored content, (3) single-shot learning from one exposure, (4) instance-specific detail preservation, and (5) contextual binding — who, when, where, why. Current LLM agents fail at all five simultaneously. Your existing text-snippet store is attempting to serve episodic memory but lacks the temporal binding (contextual property #5) that makes episodic retrieval meaningful.

**Semantic Memory**  
De-contextualized factual knowledge: user preferences, established facts, relationship structures. This is what your Haiku extraction pipeline produces when it extracts "user prefers TypeScript over JavaScript" — that is a semantic memory. Unlike episodic memory, semantic memories have no timestamp anchor and degrade in importance when contradicted by newer facts, not when they age.

**Procedural Memory**  
How-to knowledge: workflows, patterns of action, agent-specific behaviors. CoALA notes this can be "embedded in agent code or LLM parameters." LangMem explicitly supports this as a third memory type alongside semantic and episodic. For Recall, this maps to stored procedures, workflow templates, or decision patterns — currently not supported.

### The MemGPT Hierarchy (Main Context vs. External Context)

MemGPT (Packer et al., 2023, arXiv:2310.08560) treats the LLM like a CPU and proposes a tiered memory architecture analogous to OS virtual memory:

| Tier | Analogy | Content | Access Pattern |
|------|---------|---------|----------------|
| Main Context | RAM | Active conversation, current task, core persona | Always in-context; instant |
| Core Memory | Fast disk cache | Stable user profile, agent persona, key facts | Kept in-context, rarely evicted |
| Recall Storage | HDD | Indexed conversation history | Search-retrieved on demand |
| Archival Storage | Cold storage | Full documents, large reference data | Search-retrieved, rarely needed |

The key mechanism is **virtual context management**: when main context approaches token limit, the agent uses function calls to page information in/out of external tiers. This is analogous to OS memory paging. Recall maps most directly to MemGPT's "Recall Storage" tier — indexed, search-retrieved conversation history.

### What Should Be Stored: Raw Text vs. Structured vs. Embeddings

The **"Rethinking Memory in AI" survey** (Du et al., 2025, arXiv:2505.00675) identifies three representational substrates:

**Contextual Unstructured Memory** — raw text, summaries, dialogue turns. Flexible and universal but loses relational structure. Good for episodic records. Your current approach.

**Contextual Structured Memory** — knowledge graphs, relational tables, JSON schemas. Preserves relationships, enables precise querying, supports symbolic reasoning. Required for entity/relationship queries ("what does the user know about project X?"). Missing from Recall v0.1.

**Parametric Memory** — knowledge baked into model weights via fine-tuning. Not relevant for a runtime memory layer; this is a training-time concern.

The 2026 survey "Memory for Autonomous LLM Agents" (arXiv:2603.07670) concludes that production systems use all three: "most effective systems blend at least two representational substrates, using vector-indexed stores for semantic similarity search and structured stores for entity and temporal queries."

### Memory Representation in Practice

**A-MEM** (Xu et al., 2025, arXiv:2502.12110) stores seven fields per memory note:
- `c_i` — original interaction content
- `t_i` — timestamp
- `K_i` — LLM-generated keywords
- `G_i` — LLM-generated categorical tags  
- `X_i` — LLM-generated contextual description
- `e_i` — dense vector embedding
- `L_i` — set of linked memory IDs

This is the most complete single-memory schema in the literature. The key addition over Recall's current schema is the explicit `L_i` linkage field — memories point to related memories, enabling network traversal on retrieval.

---

## 2. Retrieval Math — What Actually Works

### BM25 vs. Dense Retrieval vs. Hybrid: The Verdict

The literature is unambiguous: **hybrid retrieval consistently outperforms either method alone**.

From a 2026 financial QA benchmark (arXiv:2604.01733):

| Method | Recall@5 |
|--------|----------|
| Dense (embeddings only) | 0.587 |
| BM25 only | 0.644 |
| Hybrid RRF | 0.695 |
| Hybrid + neural reranker | 0.816 |

BEIR benchmark averages (2024–2025 meta-analysis): BM25 nDCG@10 = 43.4, hybrid with reranking = 52.6+. For open-domain QA: hybrid pipelines achieve 53.4% passage recall vs. BM25's 22.1% and dense-only's 48.7%.

**Why each method fails alone:**
- Dense/embedding retrieval: excellent semantic coverage, misses exact keywords, domain-specific terms, proper nouns, version numbers, error codes
- BM25: exact lexical matching, misses paraphrase, synonym variation, semantic drift
- Hybrid: captures both failure modes

IBM research (2024) found that **three-way retrieval** (BM25 + dense embeddings + sparse learned vectors like SPLADE) is the optimal configuration, though the marginal gain from adding the third leg is smaller than the first hybrid step. Adding BM25 + dense already captures most of the benefit.

**Running two retrievers in parallel adds approximately 6ms to p50 latency** — acceptable for async memory systems.

### Reciprocal Rank Fusion (RRF) — Exact Formula

RRF is the standard fusion method across the literature. From Cormack et al. (2009), applied uniformly in agent memory systems:

```
RRF_score(d) = Σ_{r ∈ R}  1 / (k + rank_r(d))
```

Where:
- `d` = document/memory being scored
- `R` = set of ranker systems (e.g., {BM25_ranker, dense_ranker})
- `rank_r(d)` = rank position of `d` in ranker `r` (1-indexed)
- `k` = smoothing constant, empirically tuned to **60** across most benchmarks

The constant `k = 60` prevents top-ranked documents from dominating and provides robustness when one ranker returns irrelevant results. Documents not present in a ranker's results receive a score of 0 from that ranker.

**Weighted RRF** (used in Zep and MAGMA):
```
WRRF_score(d) = Σ_{r ∈ R}  w_r / (k + rank_r(d))
```

Where `w_r` is the weight assigned to ranker `r`. Default `w_r = 1.0` gives standard RRF. In practice, domain-specific tuning of weights per query type improves performance: BM25-heavy for entity/keyword queries, embedding-heavy for semantic queries.

### Recency Scoring: Exponential Decay vs. Power Law

**Exponential decay** is the dominant approach in deployed systems. The Stanford Generative Agents paper (Park et al., 2023, arXiv:2304.03442) established the canonical formula:

```
recency(m) = decay_factor ^ (t_now - t_last_accessed)
```

With `decay_factor = 0.995` per hour in their simulation. For real-world use:
- Short-memory sessions (hours): `decay_factor = 0.99` (per-minute unit)
- Medium-term (days): `decay_factor = 0.98` (per-hour unit)
- Long-term (weeks): `decay_factor = 0.95` (per-day unit)

**MemoryBank's Ebbinghaus-inspired approach** uses:
```
R = e^(-t/S)
```

Where `R` is retention (0–1), `t` is time elapsed since last access, and `S` is memory strength. `S` initializes to 1 at first encoding, increments by 1 on each retrieval, and resets the time counter `t`. This means frequently-accessed memories have a much flatter decay curve — exactly the spacing effect from human memory research.

**Power law vs. exponential**: Research by Kahana (2002) on human memory shows that individual memory traces decay exponentially, but aggregate forgetting curves appear as power laws due to variance in individual `S` values. For a computational system, exponential decay with strength-based reinforcement (the MemoryBank approach) is both theoretically grounded and computationally tractable.

**ACT-R base-level learning equation** (Human-Like Remembering paper, HAI 2024) provides the most cognitively grounded decay formula:

```
A(i) = ln(Σ_{j=1}^{n} t_j^{-d}) + β_A
```

Where:
- `A(i)` = activation strength of memory `i`
- `n` = number of past retrievals
- `t_j` = time elapsed since the j-th retrieval
- `d` = decay parameter ≈ 0.5 (empirically validated)
- `β_A` = base noise constant

This is computationally heavier than simple exponential decay but provides human-like behavior: recent memories are highly active, repeatedly accessed memories decay more slowly, and never-accessed memories fade to near-zero. Experiments confirmed this model reproduces both memory reinforcement through repetition and stochastic retrieval variability.

### The Combined Ranking Formula

The **Generative Agents** paper (Park et al., 2023) established the three-component model that has become the field consensus:

```
score(m) = α_recency · recency(m) + α_importance · importance(m) + α_relevance · relevance(m)
```

All terms normalized to [0, 1] via min-max scaling over the candidate set. The original paper used `α_recency = α_importance = α_relevance = 1.0` (equal weights), then selected top-k by final score.

**LangMem** extends this with strength (access frequency): "Memory relevance is more than semantic similarity. Recall should combine similarity with importance of the memory, as well as memory strength, which is a function of how recently/frequently it was used."

**Practical formula for Recall:**
```
final_score(m) = w1 · hybrid_relevance(m, q)   # RRF of BM25 + cosine
              + w2 · recency(m)                  # e^(-t/S) or decay^Δt  
              + w3 · importance(m)               # stored 0-1 float
              + w4 · access_frequency(m)         # log(1 + access_count)
```

Where weights `w1=0.5, w2=0.2, w3=0.2, w4=0.1` are reasonable defaults (tune per domain). Use min-max normalization within each component before combining.

### Does Importance Weighting Actually Help?

Yes, with caveats. The Generative Agents paper validates that importance-based ranking separates mundane from significant memories. However, the quality of the 1–10 importance score matters enormously.

The standard method is LLM-assigned poignancy:
> "On a scale of 1 to 10, where 1 is purely mundane (brushing teeth) and 10 is extremely poignant (a breakup, major decision), rate the likely significance of this memory."

Your Haiku extraction pipeline already assigns an importance float — this is the right approach. The research confirms it helps, but the gain is modest (+5–10% retrieval quality) compared to the larger gain from hybrid retrieval (BM25 + dense).

---

## 3. What Queries Agents Actually Make

### The Four Query Classes

Production agent memory systems see four structurally distinct query patterns. Pure semantic similarity search handles only one adequately.

**Class 1: Semantic Similarity Queries**
"What does this user prefer about database design?"  
Retrieval method: Dense embedding cosine similarity. BM25 adds marginal value here. Standard RAG covers this case.

**Class 2: Temporal/Sequential Queries**
"What did we decide about authentication last week?"  
"What was the most recent conversation about project X?"  
Retrieval method: Temporal filtering + recency ranking. Vector similarity alone fails catastrophically here — a semantically similar memory from six months ago outranks a less-similar but recent one. Requires timestamps and temporal filtering as first-class retrieval primitives, not post-hoc filters.

**Class 3: Entity/Relationship Queries**
"Everything about the user's database stack."  
"What do I know about team Y's deployment process?"  
Retrieval method: Entity graph traversal. BM25 partially handles this for named entities. Full entity queries require structured storage — an entity extracted as a node in a graph, with all related memories linked. MAGMA's entity graph and Zep's entity types both address this. Pure vector search fails at multi-hop: "find all memories connected to entity X through relationship R."

**Class 4: Contradiction/Consistency Queries**
"Do I have conflicting information about the user's preferred stack?"  
"Is there newer information that supersedes this memory?"  
Retrieval method: This is not a standard retrieval query — it requires active contradiction detection during storage or a dedicated consistency sweep. No existing system does this well. The 2026 survey calls contradiction detection "necessary but underdeveloped."

### How Agent Memory Retrieval Differs From Document RAG

The "Episodic Memory is the Missing Piece" paper (arXiv:2502.06975) articulates the core distinction:

- **Document RAG**: Retrieve chunks based on semantic similarity from a heterogeneous corpus. The chunks were written to be retrieved; they contain self-contained information.
- **Agent episodic memory**: Retrieve events from a coherent dialogue stream. The entries are not self-contained — they reference shared context (pronouns, ellipsis, running decisions). Retrieval must handle coreference resolution, topic drift, and temporal dependencies that don't exist in document corpora.

The MemReranker paper (arXiv:2605.06132) provides empirical evidence: "Generic reranking models rely heavily on semantic similarity, but semantic similarity does not equate to containing the answer." On the LOCOMO benchmark, pure semantic retrieval achieves 0.58 MAP; reasoning-aware reranking achieves 0.74 MAP. The gap is largest for temporal and causal queries.

**Practical implication for Recall**: BM25 alone (your current approach) handles Class 1 queries adequately but degrades significantly on Class 2 (temporal) and fails on Class 3 (entity) and Class 4 (contradiction). Adding dense retrieval with RRF fusion addresses Class 1 and partially Class 2. Classes 3 and 4 require structural additions.

---

## 4. What Data the System Should Accept

### Current Recall Limitation

Recall v0.1 accepts plain text conversation strings. The research shows this is insufficient for production agent memory in four specific ways:

### What Production Systems Accept

**Structured Facts (Key-Value Pairs)**  
Zep's entity types (2025) accept Pydantic-style schemas for domain-specific entities:
```python
class UserPreference(ZepEntityType):
    category: str      # "database", "language", "tooling"
    preference: str    # "PostgreSQL over MySQL"
    confidence: float  # 0.0-1.0
    context: str       # "mentioned during schema discussion"
```

Mem0 extracts structured facts from conversations into its vector store alongside raw embeddings. LangMem's MemoryManager converts "I love PostgreSQL" into `{"entity": "user", "attribute": "database_preference", "value": "PostgreSQL"}`.

The research consensus: structured extraction during ingestion costs one extra LLM call but makes entity queries and contradiction detection tractable. Your Haiku extraction pipeline already does this — the gap is that the extracted facts should be stored in a queryable structured format, not just embedded as text.

**Code Snippets**  
MAGMA (arXiv:2601.03236) and the "Intrinsic Memory Agents" paper (arXiv:2508.08997) handle heterogeneous input including code. Code requires special handling: embedding models trained on natural language perform poorly on code (semantic clustering of code requires code-specific embeddings like CodeBERT or text-embedding-3-large, which handles code reasonably well). BM25 over code is actually effective for token-level matching. Code memories also have special staleness semantics — "the auth middleware was written like this three months ago" needs a version/timestamp tag more than other memory types.

**Decisions with Context**  
The literature distinguishes "decisions" as a specific memory subtype. A decision memory requires:
- The decision itself (what was decided)
- The rationale (why)
- The constraints at decision time (context)
- The timestamp (when)
- Who was party to the decision (actors)

Zep models these as fact edges with temporal validity windows: "decision(use_postgres) valid from 2024-11 superseded by decision(migrate_to_supabase) from 2025-03." This directly addresses the contradiction problem: a newer decision automatically invalidates the older one.

**Entity/Relationship Data**  
This is the gap that graph-based systems (Zep, MAGMA, A-MEM) address that pure vector stores cannot. When a user mentions "our auth team," the memory system should extract:
- Entity: `auth_team` (type: team)
- Entity: `user` (type: person)
- Relationship: `user is_member_of auth_team`

Subsequent queries about "anything related to auth" then traverse this graph rather than relying on semantic similarity to accidentally retrieve the right memory.

### The Heterogeneous Memory Case

The "Intrinsic Memory Agents" paper (arXiv:2508.08997) explicitly argues that homogeneous memory (everything stored as flat text embeddings) "decreases benefits of having agents focused on single parts of tasks." Different input types need different storage representations:

| Input Type | Best Storage | Retrieval Method |
|------------|-------------|-----------------|
| Conversation turn | Raw text + embedding | Semantic search + BM25 |
| Extracted fact | Structured JSON + embedding | Exact match + semantic |
| Code snippet | Raw text + code embedding | BM25 (token match) + semantic |
| Decision | Structured + temporal edge | Temporal filter + semantic |
| Entity/relationship | Graph node/edge | Graph traversal |
| Procedure | Structured + embedding | Semantic + exact type filter |

---

## 5. Memory Operations Beyond Store/Retrieve

### Contradiction Detection

This is the most underdeveloped operation in current systems. The "Rethinking Memory" survey (arXiv:2505.00675) defines contradictions as "factual or semantic inconsistencies that arise during retrieval and reasoning over heterogeneous memory representations."

**What the research recommends:**

The simplest approach is **temporal supersession**: when a new memory is ingested, retrieve top-k similar memories and check for semantic negation. Zep implements bitemporal edges — each fact has a `(t_valid, t_invalid)` window. When a contradicting fact is extracted, the old edge's `t_invalid` is set to now. The old memory is not deleted; it becomes a historical record.

Approaches by sophistication:

1. **Temporal ordering (lowest bar)**: The newest fact about X supersedes older facts about X. Requires only timestamp comparison. Misses non-temporal contradictions.

2. **Embedding similarity + temporal**: During ingestion, retrieve memories with cosine similarity > threshold to the new memory, then use LLM to check if they conflict. O(retrieval) cost per ingestion.

3. **Knowledge graph consistency**: For structured facts, check `(entity, attribute)` uniqueness. "user prefers PostgreSQL" and "user prefers MySQL" are contradictory because same entity + attribute with different values. This is checkable without LLM.

4. **Full semantic contradiction checking**: LLM-based entailment checking against top-k retrieved memories. Most reliable, most expensive. Research shows this works but is rarely implemented at scale.

**Practical recommendation for Recall**: Implement approach #3 for extracted structured facts (cheap, reliable for your existing Haiku extraction output) and approach #2 for raw text memories (affordable with Haiku at ingestion time). Flag contradictions with a `superseded_by` pointer rather than deleting — historical memory has value.

### Memory Consolidation

The literature uniformly agrees consolidation is "the most important and least implemented" operation (2025 survey on LongDialQA). Three consolidation patterns:

**Episodic → Semantic**: Many raw conversation turns about the same topic get summarized into a single semantic fact. Example: 20 turns about database preferences become "user strongly prefers PostgreSQL, has worked with it for 5 years, has used it in production with Supabase." This prevents context bloat from repeated similar memories degrading retrieval quality.

**Rolling summary**: Periodic consolidation of the N oldest episodic memories into a summary. Used in MemGPT and most sliding-window approaches. Risk: information loss — the 2025 survey estimates naive summarization loses ~20% of encoded facts.

**Event-triggered consolidation** (GAM, arXiv:2604.12285): Consolidate when a semantic shift is detected, not on a fixed schedule. The GAM framework uses a "Semantic-Event-Triggered" mechanism: dialogue is stored in a local event progression graph, and only when topic shift is detected does it consolidate into the global topic associative network. This minimizes interference from transient context while preserving long-term consistency.

**Consolidation frequency**: The 2025 survey recommends every 50–200 episodes as a practical range for scheduled consolidation. Too frequent = information loss from premature summarization; too infrequent = retrieval quality degradation from fragmented episodic records.

### Memory Decay

The research is split on whether decay improves agent performance in practice:

**Arguments for decay:**
- Prevents stale memories from polluting retrieval (temporal obsolescence)
- Storage efficiency
- Privacy and compliance (GDPR right-to-be-forgotten)
- Mirrors human memory for more natural interactions

**Arguments against automatic decay:**
- Loss of valuable historical context
- Difficult to set the right decay rate without domain knowledge
- A memory not accessed for a year may still be critical

**What production systems do:** Mem0 (as of 2025) uses explicit TTL for specific memory categories rather than universal decay. Zep uses temporal validity windows (`t_invalid`) rather than decay scores. The Ebbinghaus approach (MemoryBank) is the most principled but most complex.

**Research consensus (2025)**: Hard TTL-based expiration is too crude. Strength-based decay (MemoryBank's `S` counter) provides better discrimination. But the most important "forgetting" is contradiction resolution — superseding stale facts is more valuable than degrading their scores. The 2026 survey ("Memory for Autonomous LLM Agents") states: "nobody evaluates forgetting well, with only one benchmark testing selective forgetting explicitly."

### Memory Summarization (Episodic → Semantic Compression)

When to compress episodic to semantic:
- On topic shift (GAM's event-triggered approach)
- When episodic store size exceeds threshold
- On session end (create a session summary)
- Periodically by schedule

What to preserve in summaries:
- Key entities and their attributes
- Decisions made and their rationale
- User preferences and corrections
- Unresolved questions or open loops

What compression loses:
- Exact phrasing (may matter for instruction-following)
- Temporal sequence within the topic
- Uncertainty markers ("I think" vs. "definitely")

The GAM paper's practical design — buffer recent episodes in an event graph, consolidate to topic network on semantic shift — is the most validated approach in the literature.

---

## 6. Paper Summaries

### MemGPT: Towards LLMs as Operating Systems
**Citation:** Packer, C. et al. (2023). arXiv:2310.08560  
**Key Insight:** Treats LLM context window as RAM and proposes virtual context management — paging memory in/out of a tiered hierarchy (main context → core → recall → archival) analogous to OS virtual memory. The agent uses function calls to explicitly manage what is in context. DMR benchmark: 93.4% accuracy on deep memory retrieval.  
**Relevant Math:** No explicit scoring formula for retrieval; uses BM25 + embedding search over recall/archival storage.  
**Implication for Recall:** Your system is Recall Storage in MemGPT's hierarchy. The key lesson is that the agent should be able to explicitly request what it needs ("search for X") rather than relying on passive retrieval injection. Support agent-initiated retrieval, not just application-initiated.

---

### A-MEM: Agentic Memory for LLM Agents
**Citation:** Xu, W. et al. (2025). arXiv:2502.12110. NeurIPS 2025.  
**Key Insight:** Inspired by Zettelkasten note-taking. Each memory is stored as a 7-field note: content, timestamp, keywords, tags, contextual description, embedding, and a link set. The system dynamically links memories during ingestion (using cosine similarity + LLM connection analysis) and updates historical memory attributes when new memories integrate. The network structure enables linked memories to be co-retrieved.  
**Relevant Math:** Cosine similarity for candidate link discovery: `s_{n,j} = (e_n · e_j) / (|e_n| |e_j|)`. Top-k candidates then assessed by LLM for semantic/causal links.  
**Performance:** 2x improvement on multi-hop reasoning (LoCoMo), 35% F1 improvement on DialSim, 85–93% token reduction vs. baselines. Scales from 1K to 1M memories with retrieval time 0.31μs → 3.70μs.  
**Implication for Recall:** Add explicit link fields to your memory schema. Link-based co-retrieval (when you retrieve memory A, also fetch its linked memories B, C) dramatically improves multi-hop query quality. LLM-generated keywords and contextual descriptions are justified at this scale.

---

### MemoryBank: Enhancing LLMs with Long-Term Memory
**Citation:** Zhong, W. et al. (2023). arXiv:2305.10250. AAAI 2024.  
**Key Insight:** Applies Ebbinghaus forgetting curve to control memory retention. Retention formula: `R = e^(-t/S)` where `t` is elapsed time and `S` is memory strength initialized to 1, incremented on each retrieval. Retrieved memories are reinforced (S++ and t reset); unaccessed memories fade. Demonstrated on SiliconFriend companion chatbot.  
**Relevant Math:** `R = e^(-t/S)`. Caveat: the authors acknowledge this is "exploratory and highly simplified."  
**Implication for Recall:** The strength counter (your `access_count` field, if you have one) is more principled than time-only decay. Track `access_count` and `last_accessed_at` per memory. Use these to compute a reinforcement-adjusted recency score rather than pure time-based decay.

---

### Cognitive Architectures for Language Agents (CoALA)
**Citation:** Sumers, T. et al. (2023). arXiv:2309.02427. TMLR February 2024.  
**Key Insight:** Formalizes the four-memory taxonomy (working, episodic, semantic, procedural) for LLM agents, grounded in cognitive science. Defines memory-organized agent architectures: each decision cycle involves retrieval from long-term memory into working memory, then planning, then execution that may write back to long-term memory. This cycle is the correct mental model for designing Recall's API.  
**Implication for Recall:** The four types require different APIs. Episodic queries need temporal context; semantic queries need entity context; procedural queries need type filters. A single "search text" API serves none of them optimally.

---

### Generative Agents: Interactive Simulacra of Human Behavior
**Citation:** Park, J.S. et al. (2023). arXiv:2304.03442. UIST 2023.  
**Key Insight:** Established the three-component retrieval scoring formula that is now the field standard: `score = α_recency · recency + α_importance · importance + α_relevance · relevance`, with equal weights and min-max normalization. Recency uses exponential decay with `decay_factor = 0.995` per hour. Importance is LLM-assigned on a 1–10 "poignancy" scale. Relevance is cosine similarity of embeddings.  
**Relevant Math:** See Section 2 above for complete formulas.  
**Implication for Recall:** Your importance float is validated. Add exponential decay-based recency scoring. Normalize all three components before combining.

---

### Zep: A Temporal Knowledge Graph Architecture for Agent Memory
**Citation:** Rasmussen, P. et al. (2025). arXiv:2501.13956.  
**Key Insight:** Builds a temporally-aware knowledge graph with three subgraphs: episode (raw input), semantic entity (extracted entities and relationships), and community (clustered entity summaries). Uses bitemporal modeling: every fact carries `(t_valid, t_invalid)` and `(t'_created, t'_expired)` — separating when the fact was true in the world from when the system learned about it. This enables precise temporal reasoning and automatic contradiction resolution (new fact sets `t_invalid` on old contradicting fact).  
**Relevant Math:** Hybrid retrieval: `φ = φ_cos ∪ φ_bm25 ∪ φ_bfs`, then RRF reranking. Three-way search: cosine similarity (1024-dim embeddings), BM25 full-text, and breadth-first graph traversal.  
**Performance:** 94.8% on DMR benchmark vs. MemGPT's 93.4%; 18.5% accuracy improvement on LongMemEval with 90% latency reduction.  
**Implication for Recall:** Bitemporal fact modeling is the most principled approach to contradiction resolution. Even without building a full graph, adding `(valid_from, valid_until)` to extracted structured facts gives temporal supersession for free.

---

### Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
**Citation:** Chhikara, P. et al. (2025). arXiv:2504.19413. ECAI 2025.  
**Key Insight:** Multi-signal retrieval fusing semantic search, BM25 keyword matching, and entity extraction — results scored in parallel and fused. Token-efficient: 91% lower p95 latency and 90% token cost reduction vs. full-context approaches. Graph-enhanced variant (Mem0g) adds relational structure for complex queries. Selective memory extraction averages under 7,000 tokens per retrieval call vs. 25,000+ for full-context.  
**Performance:** LoCoMo: 91.6, LongMemEval: 93.4.  
**Implication for Recall:** The token efficiency argument is strong — your users will care about this. Selective retrieval (retrieve what's needed, not everything) requires good importance + relevance scoring. Mem0's async memory write (don't block the response) is a key production pattern.

---

### MAGMA: A Multi-Graph based Agentic Memory Architecture
**Citation:** (2025). arXiv:2601.03236.  
**Key Insight:** Stores each memory event across four orthogonal graph layers: temporal (chronological ordering), causal (logical entailment), semantic (embedding similarity), and entity (object permanence). Query intent is classified (Why/When/Entity) and routed to the appropriate graph layer with intent-aware edge weights.  
**Relevant Math:** Dynamic transition score: `S(n_j|n_i, q) = exp(λ1·φ(type(e_ij), T_q) + λ2·sim(n_j, q))` where `φ` weights edge type by query intent and `sim` is cosine similarity. RRF used for anchor node identification.  
**Implication for Recall:** Query intent classification before retrieval is valuable — different query types should use different retrieval weights. Simple intent detection (keyword-based: "when did..." → temporal, "why did..." → causal, entity names → entity) can route without an LLM.

---

### MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval
**Citation:** (2025). arXiv:2605.06132. [Published ~May 2026]  
**Key Insight:** Documents that are semantically similar to a query do not necessarily contain the answer. Generic rerankers fail on temporal, causal, and multi-hop memory queries. MemReranker (0.6B/4B) trained via LLM knowledge distillation achieves GPT-4o-mini-level reranking at 8x faster inference. Uses five-level relevance scale, handles coreference and topic drift.  
**Performance:** LoCoMo MAP: 0.737 (vs. BGE-Reranker 0.671, GPT-4o-mini 0.715). LongMemEval MAP: 0.804 (vs. Gemini-3-Flash 0.726).  
**Implication for Recall:** A retrieve-then-rerank pipeline is significantly better than scoring-at-retrieval-time. Even a small reranker (0.6B) outperforms large general-purpose models on memory-specific tasks. Adding a reranking stage is the highest-leverage single improvement for retrieval quality.

---

### Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents
**Citation:** Pink, M. et al. (2025). arXiv:2502.06975.  
**Key Insight:** Current approaches fragment episodic memory's five required properties across incompatible systems. In-context memory satisfies single-shot learning but not long-term retention. External memory (RAG) provides retention but loses contextual binding. Parametric memory provides persistence but cannot learn in-session. The paper argues for an integrated episodic system satisfying all five properties simultaneously.  
**Implication for Recall:** Your system attempts episodic storage but lacks the contextual binding properties (who/when/where/why bound to the content). Adding structured context fields at ingestion time (session ID, agent ID, task context, turn number within session) satisfies the contextual property.

---

### GAM: Hierarchical Graph-based Agentic Memory
**Citation:** Wu, Z. et al. (2025). arXiv:2604.12285. [April 2025]  
**Key Insight:** Separates memory into a local Event Progression Graph (short-term, recent context) and a global Topic Associative Network (long-term, consolidated knowledge). Consolidation is triggered by semantic shift detection rather than fixed schedules, preventing interference from transient context while enabling stable long-term knowledge.  
**Implication for Recall:** Session-scoped vs. long-term memory distinction. New memories should first go into a session buffer; consolidation to the shared long-term store should happen at session end or on topic shift — not immediately on every write.

---

### Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions
**Citation:** Du, Y. et al. (2025). arXiv:2505.00675. May 2025.  
**Key Insight:** Defines six fundamental memory operations: Consolidation, Indexing, Updating, Forgetting, Retrieval, and Compression. Categorizes memory as parametric (in-weights), contextual unstructured, or contextual structured. Identifies contradiction resolution across heterogeneous sources as the primary open research challenge.  
**Implication for Recall:** Your current implementation covers Indexing (BM25) and Retrieval. Missing: Consolidation, Updating (contradiction resolution), Forgetting (decay/TTL), and Compression (summarization). These are v0.2 targets.

---

## 7. Design Implications for Recall v0.2

This section synthesizes what the research says Recall should change and add, ordered by impact.

### Priority 1: Hybrid Retrieval (Highest Impact, Moderate Effort)

**What:** Add dense embedding retrieval alongside BM25. Fuse results with RRF (k=60).  
**Why:** Hybrid retrieval improves recall by 17–39% over BM25 alone across benchmarks. Adding dense retrieval covers the semantic query class that BM25 misses (paraphrase, synonym variation).  
**Formula:**
```
RRF_score(m) = 1/(60 + rank_bm25(m)) + 1/(60 + rank_dense(m))
```

**Implementation notes:**
- Use `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (local, free) for embeddings
- Store embeddings at ingestion time alongside existing BM25 index
- Retrieve top-50 from each system, fuse with RRF, return top-k from fused results
- Parallel retrieval adds ~6ms latency — acceptable

### Priority 2: Improved Scoring Formula (High Impact, Low Effort)

**What:** Replace BM25-only score with the three-component formula.  
**Current:** Final score = BM25 score + recency_bonus (simple time subtraction, presumably)  
**New:**
```python
final_score(m) = (
    0.50 * normalize(rrf_score(m, q))   # hybrid relevance
  + 0.25 * normalize(recency(m))         # e^(-Δdays / S)
  + 0.15 * normalize(importance(m))      # stored 0-1 float from Haiku
  + 0.10 * normalize(log(1 + access_count(m)))  # strength bonus
)
```

**Recency function:**
```python
def recency(memory):
    delta_days = (now - memory.last_accessed_at).days
    S = memory.access_count or 1  # strength = 1 + number of retrievals
    return math.exp(-delta_days / S)
```

**Why this is correct:** Matches the Park et al. (2023) formula, adds MemoryBank's strength modifier, maps directly onto your existing `importance` field. Does not require new infrastructure.

### Priority 3: Add Structured Fact Storage (High Impact, Moderate Effort)

**What:** When Haiku extracts typed facts (preference, decision, procedure), store them as structured JSON records in addition to (or instead of) raw text embedding. Add a `fact_type` field, `entity` field, and `attribute` field.  
**Why:** Enables entity queries ("everything about auth"), exact-match lookups, and contradiction detection without LLM calls.  
**Schema addition:**
```json
{
  "memory_id": "uuid",
  "raw_text": "The user prefers PostgreSQL for the auth service",
  "embedding": [...],
  "fact_type": "preference",     // preference | decision | procedure | fact | event
  "entity": "user",
  "attribute": "database_preference",
  "value": "PostgreSQL",
  "context": "auth service",
  "valid_from": "2025-11-01",
  "valid_until": null,           // null = still valid
  "superseded_by": null          // UUID of newer contradicting memory
}
```

This schema directly supports contradiction resolution: when a new memory sets `valid_until` and `superseded_by`, old retrieval code can filter to `valid_until IS NULL` for current facts.

### Priority 4: Temporal Metadata on All Memories (High Impact, Low Effort)

**What:** Add session-level context fields to every memory at ingestion:
```json
{
  "session_id": "...",
  "turn_number": 14,
  "agent_id": "...",
  "created_at": "...",
  "last_accessed_at": "...",
  "access_count": 0
}
```

**Why:** Enables temporal queries ("what did we decide last week?"), session-scoped retrieval, and the recency formula above. Also satisfies the episodic contextual binding property from arXiv:2502.06975.

### Priority 5: Contradiction Detection at Ingestion (Medium Impact, Low Effort for Structured Facts)

**What:** For structured fact memories, check for existing memories with the same `(entity, attribute)` pair. If found and conflicting, set `valid_until = now` and `superseded_by = new_memory_id` on the old record.  
**Why:** This is the cheapest form of contradiction resolution — no LLM call required, just a database query on indexed fields.  
**Scope:** Only for extracted structured facts, not raw episodic text. The Zep bitemporal model validated this approach at production scale.

### Priority 6: Retrieve-Then-Rerank Pipeline (Medium Impact, Higher Effort)

**What:** After hybrid retrieval returns top-50 candidates, apply a reranking model before returning top-k.  
**Why:** MemReranker (arXiv:2605.06132) shows that semantic similarity fails for temporal and causal queries — the types agents most commonly make beyond simple preference lookups. A 0.6B reranker gets you GPT-4o-mini-level reranking at 8x lower latency.  
**Options:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (free, local, 80ms reranking for 50 candidates)
- Cohere Rerank API (hosted, ~$0.001/query at scale)
- Skip for v0.2 if latency budget is tight; add in v0.3

### Priority 7: Memory Linking (Medium Impact, Higher Effort)

**What:** Implement A-MEM's linking mechanism: at ingestion, find top-k similar existing memories by cosine similarity, use LLM to confirm meaningful connections, store link IDs on the new memory.  
**Why:** A-MEM shows 2x improvement on multi-hop reasoning and 85–93% token reduction. When memory A is retrieved, co-retrieve its linked memories B and C — the agent gets richer context without longer queries.  
**Note:** This is an enhancement, not a prerequisite. The simpler version is just: store cosine-similarity-based "related memory IDs" at ingestion, return them alongside retrieved memories so the caller can optionally fetch them.

### Priority 8: Session Buffer + Consolidation (Lower Priority, Architectural)

**What:** Introduce a session-scoped memory buffer. New memories land in the buffer. On session end, consolidate: merge highly similar memories, promote important memories to long-term store, generate a session summary.  
**Why:** GAM (arXiv:2604.12285) shows this prevents transient context noise from polluting the long-term store. LoCoMo and LongMemEval benchmark improvements.  
**Effort:** Architectural change. Requires deciding consolidation triggers and summary templates.

---

### Recommended Ranking Formula (Final)

Based on all research reviewed, here is the scoring formula Recall should target for v0.2:

```
score(m, q) = w_rel · normalize(RRF(BM25(m,q), cosine(m,q)))
            + w_rec · exp(-Δt / (1 + access_count(m)))
            + w_imp · importance(m)
            + w_str · log(1 + access_count(m)) / log(1 + max_access_count)

# Default weights: w_rel=0.50, w_rec=0.25, w_imp=0.15, w_str=0.10
```

All terms normalized to [0,1] via min-max scaling over the candidate set before combining.

**What this adds over current Recall:**
- `RRF(BM25, cosine)` — adds dense retrieval, replaces BM25-only
- `exp(-Δt / (1 + access_count))` — adds MemoryBank-style strength modifier to recency
- `access_count` in denominator of recency — frequently-accessed memories decay more slowly
- `log(1 + access_count)` strength bonus — reinforces memories that have proven useful

---

### What Data Types to Add (Summary)

| Data Type | Intake Format | Storage Change Needed |
|-----------|--------------|----------------------|
| Structured facts (current output of Haiku) | JSON with entity/attribute/value | Add structured columns alongside embedding |
| Decisions | JSON with rationale + temporal validity | Add `valid_from`, `valid_until`, `superseded_by` |
| Session context | Auto-attached at ingestion | Add `session_id`, `turn_number` to every memory |
| Procedures/workflows | Text with `fact_type=procedure` | Type-filtered retrieval |
| Code snippets | Raw text with `fact_type=code` | Use code-aware embedding or rely on BM25 |
| Linked memories | Computed at ingestion | Add `linked_memory_ids` array field |

---

## 8. Resources

**Primary Papers**

- [MemGPT: Towards LLMs as Operating Systems (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)
- [A-MEM: Agentic Memory for LLM Agents (arXiv:2502.12110)](https://arxiv.org/abs/2502.12110)
- [MemoryBank: Enhancing LLMs with Long-Term Memory (arXiv:2305.10250)](https://arxiv.org/abs/2305.10250)
- [Cognitive Architectures for Language Agents / CoALA (arXiv:2309.02427)](https://arxiv.org/abs/2309.02427)
- [Generative Agents: Interactive Simulacra of Human Behavior (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)
- [MAGMA: A Multi-Graph based Agentic Memory Architecture (arXiv:2601.03236)](https://arxiv.org/html/2601.03236v1)
- [Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents (arXiv:2502.06975)](https://arxiv.org/abs/2502.06975)
- [GAM: Hierarchical Graph-based Agentic Memory for LLM Agents (arXiv:2604.12285)](https://arxiv.org/abs/2604.12285)
- [MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval (arXiv:2605.06132)](https://arxiv.org/abs/2605.06132)
- [Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions (arXiv:2505.00675)](https://arxiv.org/abs/2505.00675)
- [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers (arXiv:2603.07670)](https://arxiv.org/html/2603.07670v1)

**Surveys**

- [A Survey on the Memory Mechanism of LLM-based Agents (ACM TOIS)](https://dl.acm.org/doi/10.1145/3748302)
- [Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems (arXiv:2508.08997)](https://arxiv.org/html/2508.08997v1)
- [Graph-based Agent Memory: Taxonomy, Techniques, and Applications (arXiv:2602.05665)](https://arxiv.org/html/2602.05665v1)

**Production Systems**

- [Zep GitHub (Graphiti)](https://github.com/getzep/graphiti)
- [Zep Entity Types Blog Post](https://blog.getzep.com/entity-types-structured-agent-memory/)
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Mem0 State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [LangMem Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)

**Retrieval Math**

- [Reciprocal Rank Fusion — Elasticsearch Reference](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [Hybrid Search Scoring (RRF) — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies (arXiv:2604.01733)](https://arxiv.org/html/2604.01733v1)

**Benchmarks**

- [LongMemEval (ICLR 2025)](https://github.com/xiaowu0162/LongMemEval)
- [LoCoMo: Evaluating Very Long-Term Conversational Memory](https://snap-research.github.io/locomo/)
- [MemBench (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.989.pdf)
