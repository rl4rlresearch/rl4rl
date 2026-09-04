# Claim–source ledger: fixation in LLM research agents

This ledger is the audit trail for `report-source.md`. “Directness” describes relevance to longitudinal executable research-agent fixation, not paper quality.

| Claim family | Sources | Directness | Boundary / caveat |
|---|---|---|---|
| Salient irrelevant associations can fixate LLM grouping | S01 | Near-direct | Puzzle grouping, not code search; WordNet manipulation also changes difficulty |
| Biased hints anchor LLM responses and generic reflection is weak | S02 | Near-direct | Preprint; short-answer task |
| Seeing one's own prior answer causally raises persistence/confidence | S03 | Direct mechanism | Stateless two-stage choice, not long-horizon agents |
| Self-refinement can amplify self-bias | S04-S05 | Near-direct | Generation/judging settings; quality confounding must be controlled |
| Multi-turn histories introduce assumptions and reliance on wrong attempts | S06-S07 | Direct mechanism | General generation tasks/conversations, not architecture search |
| Retrieved memories induce experience-following and error propagation | S08 | Direct agent mechanism | Several agent domains; exact effect depends on retrieval policy |
| Consolidated memories can degrade below no memory | S09-S10 | Direct but preliminary | 2026 preprints; requires independent replication |
| Repeated reflection stalls or echoes | S11-S12 | Near-direct | DORA tasks and RL-trained reflection differ from inference-only runs |
| Research agents narrow scientific ideation | S13 | Direct domain | Recent preprint; text ideas, not executed descendants |
| Within-run ML-agent P-creativity declines | S14 | Highly direct | Up to ten valid episodes; novelty judge partly model-based |
| AI assistance can cause fixation displacement | S15 | Mechanistic analogue | Human-AI design, not autonomous agents |
| Dense multi-agent communication can collapse diversity | S16 | Direct system mechanism | Open-ended ideation, not model compression |
| Coupled strategy/generation prompts can cause premature convergence | S17 | Near-direct | Small Bayesian-optimization study |
| Examples narrow design variety and induce feature copying | S18-S20 | Foundational analogue | Human design cognition |
| Self-generated initial ideas can cause fixation | S21 | Strong analogue | Human designers; intervention differs |
| Warnings are weaker than grounded build/test evidence for familiar fixation | S22-S23 | Strong intervention analogue | Physical design tasks |
| Interruption/distribution can reduce repeated retrieval | S24-S26 | Analogue | Human memory/design; no direct LLM causal proof |
| AI may raise individual quality while reducing collective variety | S27-S31 | Population evidence | Mixes human-AI and model-only generation |
| Heterogeneity/independence can counter population convergence | S16, S32 | Direct system evidence | Results depend on roles and topology |
| Prompt formatting can suppress semantic diversity | S33 | Direct prompt evidence | Semantic outputs; executable-mechanism transfer unproven |
| Novelty and feasibility/impact can diverge | S14, S19, S28, S34, S49 | Strong cross-domain convergence | Metrics and units differ by study |
| External feedback is more reliable than intrinsic self-correction | S22, S40-S41 | Strong | “Reliable” depends on the feedback oracle |
| Denial/challenge prompting can increase divergence | S38, P2 | Direct | Can lower feasibility and create a new attractor |
| Quality-diversity search preserves behavioral niches | S42-S44 | Algorithmic analogue | Requires valid task-specific descriptors |
| Project prompt changes immediate executable mechanisms | P2 | Exact state-matched local evidence | Fixed labels, one model/task family; no private cognition claim |
| Project effect is punctuated, then followed by exploitation | P2 | Direct longitudinal evidence | Repeated regime later confounds first-prompt-only effect |
| Project challenge displaces rather than eliminates population fixation | P2, S15, S29-S31 | Project evidence + analogues | Factorization may also be a genuinely strong task affordance |

## Source status notes

- **Peer-reviewed / archival proceedings:** S01, S03-S06, S08, S11-S12, S14-S16, S18-S33, S35, S38-S50 (except where the linked record explicitly remains arXiv-only).
- **Recent preprints requiring extra caution:** S02, S07, S09-S10, S13, S17, S34, S36-S37.
- **Distant analogy only:** S42-S45. These should motivate mechanisms or interventions, not be presented as direct evidence that LLMs share the same failure process.
- **Repository evidence:** P2 is auditable from saved prompts, candidate source, evaluator traces, and exact matched prefixes. The mechanism taxonomy is post hoc and condition-aware; primary claims should rest on source/AST/evaluator outcomes.

## Claim language guide

Use:

- “trajectory fixation,” “search lock-in,” “mechanism persistence,” or “incumbent attractor”;
- “the recorded proposal/source changed” rather than “the model thought”;
- “state-matched paired contrast” rather than randomized causal estimate;
- “new relative to recorded trajectory history” rather than globally novel;
- “qualified structural novelty” rather than creativity when usefulness is narrow.

Avoid:

- treating every repeated edit as irrational;
- treating textual assumption language as proof of mechanistic understanding;
- treating a new family tag as scientific novelty;
- claiming prompt effects generalize across models or tasks without replication;
- conflating within-run fixation, population monoculture, and recursive-training model collapse.
