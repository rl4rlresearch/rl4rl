# Related-work and contribution-boundary audit

This audit supports the manuscript's literature positioning. It is deliberately
more conservative than a priority claim: the paper does not claim to invent
reflection, assumption questioning, alternative elicitation, evolutionary code
search, or novelty measurement.

## Closest work on research-agent narrowing

- **Bhushan, Zhang, and Wang (2026), _Can LLM Agents Discover?_** measures
  within-run novelty, historical novelty, and usefulness on ML-engineering
  tasks and reports declining novelty during exploitation. It establishes the
  longitudinal narrowing phenomenon. It does not repeatedly intervene within
  the same executable trajectory or connect the prompted departure to exact
  patches, retention decisions, and ordinary descendants.
- **Tang and Yang (2026), _AI Research Agents Narrow Scientific
  Exploration_** compares 37,802 research-agent ideas with human follow-on
  work and finds concentration near seed literature. It establishes
  population-level narrowing in textual ideas. It does not study an
  evaluator-driven code-search trajectory or a scheduled within-run treatment.
- **Ning et al. (2026), _Auto Research with Specialist Agents Develops
  Effective and Non-Trivial Training Recipes_** records hypotheses, code
  changes, evaluator outcomes, and failure types in an auditable research
  system. It is the closest systems precedent for trace granularity. It does
  not contrast ordinary and scheduled assumption-challenge policies or measure
  descendant and population responses to those checkpoints.
- **Antoniades et al. (2026), _Heuresis_** is the closest systems-level
  comparator. It holds an autonomous ML-research loop fixed while comparing
  six greedy, archive, evolutionary, and divergent strategies across quality,
  diversity, and novelty. The present paper does not imply that search-policy
  diversity is unstudied. Its distinction is a repeated semantic intervention,
  local executable contrasts, and exact ancestry rather than a comparison
  among search architectures.
- **Gumma et al. (2026), _IDEAgent_** manages research-idea lineages with
  quality-diversity search. It makes lineage and archive management explicit
  for textual ideation. The present four-lineage condition is much simpler and
  only a moderator; the empirical focus is scheduled semantic redirection in
  executable model development.

## Closest work on eliciting alternatives

- **Lu et al. (2025), _Benchmarking Language Model Creativity_** uses denial
  prompting, progressively forbidding techniques already used in code
  generation. This is a close prompt-level precedent for forcing alternatives.
  It specifies what may no longer be used and primarily evaluates generated
  solutions; the present treatment leaves the alternative unspecified and
  observes its life in a continuing empirical search process.
- **Wang (2026), _FirstResearch_** structures research-question formation
  around assumptions, mechanisms, falsifiers, and update rules. It supports the
  value of explicit assumptions but operates at question formation rather than
  as a repeated intervention in executable model development.
- **Ueda et al. (2025)** varies roles, dialogue depth, and critics in multi-agent
  research ideation. It studies dialogue architecture and idea quality rather
  than source artifacts, strict retention, or descendants.
- **Luo et al. (2026), _Inducing Sustained Creativity and Diversity in Large
  Language Models_** addresses long search quests and notes that a uniform type
  of creativity can yield homogeneous alternatives. It motivates auditing
  population conclusions under more than one semantic representation.
- **Zhou et al. (2026), _Explore Before Committing_ / HypoSearch** combats
  early trajectory commitment with bounded independent hypotheses and
  branch-level evidence in deep-research agents. It is a close precedent for
  branch management, but studies tool-mediated answer research rather than
  scheduled code interventions and strict ML evaluators.
- Reflection and self-correction work asks models to critique or repair their
  answers. The present intervention is not a correctness repair: it redirects
  the search operator before a new proposal, and an external evaluator—not the
  model's rhetoric—determines feasibility and retention.

## Closest work on fixation, memory, and diversity

- Design-fixation research (Jansson and Smith, 1991; Sio et al., 2015; Crilly
  and Cardoso, 2017) shows that examples can constrain variety while sometimes
  improving depth or quality. This motivates separating diversity from value.
- Wadinambiarachchi et al. (2024) finds that generative-AI suggestions can
  displace fixation from an initial example toward the generated replacement.
  This is the closest conceptual warning against inferring broad exploration
  merely because a trajectory leaves its prior direction.
- Alavi Naeini et al. (2023) experimentally plants red herrings that fixate
  LLMs. It demonstrates induced fixation, but not search-policy intervention in
  a long-running coding loop.
- Xiong et al. (2026) studies experience-following memory; Laban et al. (2026)
  studies early-assumption persistence in multi-turn interaction; Wu et al.
  (2025) studies generative monoculture; Chen et al. (2026) studies diversity
  collapse under multi-agent coupling. The present trajectories are independent
  rather than interacting. The full-rationale family metric initially suggests
  convergence, but mechanism-only and primary-family sensitivities reverse the
  direction. The paper therefore treats population diversity as
  measurement-dependent rather than claiming a distinct convergence mechanism.

## What is and is not contributed

The defensible contribution is the **combination of intervention, observation,
and longitudinal credit assignment**:

1. a scheduled, open-ended assumption challenge is inserted into an ongoing
   executable ML-research loop;
2. the local transition is measured against a matched ordinary transition at
   six layers—message, source artifact, feasibility, retention, cost, and
   immediate objective gain;
3. the following ordinary proposals are separated into policy-window outcomes
   and recursively traced exact descendants rather than automatically granting
   lineage credit; and
4. within-run departure is compared with between-run dispersion under several
   semantic representations, exposing construct-dependent disagreement.

No single component is claimed as historically unprecedented. The manuscript
uses the related-work section, design description, and discussion to expose the
combination and its empirical consequences; it does not include a standalone
"novelty" section.

## Claims intentionally avoided

- “The first assumption-challenge prompt.”
- “The first demonstration that LLMs fixate.”
- “The intervention increases creativity.”
- “Portfolio memory solves fixation.”
- “The intervention causally improves endpoint task performance.”
- “Broad mechanism-family labels measure scientific novelty.”
- “Challenged populations converge around a new family.”
- “All policy-window follow-up gain belongs to the challenged candidate.”

The fixed condition labels and divergent pre-intervention states make the
strongest causal wording inappropriate. The paper instead emphasizes local
matched changes, block stability, placebo transitions, trace-grounded examples,
and descriptive descendant evidence.
