# Paper 6.1 section expansion and evidence bank

This document mirrors every main-text section and subsection of **Question the
Premise, Pay the Price**. It intentionally contains
more evidence than the paper: alternative interpretations, extra quantitative
results, trace-level examples, and possible sentences that a human author can
select, rewrite, or omit. Internal condition and run names are included only
where they help locate the source record.

## Abstract: additional options

1. The intervention is unusually measurable because every public claim is
   linked to an exact patch, a fresh evaluator run, a retention decision, and a
   descendant lineage. Many prompt studies stop at ratings of text; this one can
   distinguish rhetorical novelty, structural novelty, execution, usefulness,
   and downstream use.
2. The strongest cross-task regularity is not endpoint quality. It is a process
   trade-off: more within-run novelty and larger proposals, several thousand
   extra output tokens, and lower immediate retention. That pattern survives
   the very different meanings of success across parameter-constrained
   arithmetic, image classification, and fixed-time language modeling.
3. The most useful measurement result is disagreement: full-rationale family
   tags suggest lower population dispersion, while mechanism-only and
   primary-family codings show higher challenged dispersion in every task.
   The paper improves by reporting this reversal instead of naming convergence.

An alternative abstract emphasis could lead with the follow-up-window result:
Fashion-MNIST obtains roughly 78% of its challenged-minus-control cycle gain in
the nine ordinary proposals after the challenge, and nanoGPT goes from a
negative immediate difference to a positive ten-proposal-cycle difference.
Exact ancestry then supplies the necessary correction: those gains are not all
descendant credit, especially under four-lineage rotation.

## 1. Introduction: additional evidence and angles

1. The ordinary controls are not strawmen. Their prompts say to choose the most
   informative next change from visible technical evidence, and the controls
   regularly make sophisticated, valid improvements. On addition, ordinary
   agents discover exact LayerNorm-null gauges and remove parameters one
   coordinate at a time. The paper is therefore comparing two plausible search
   policies—evidence-led exploitation and scheduled semantic redirection—not a
   thoughtful prompt against a content-free baseline.
2. “Fixation” should be treated at multiple levels. A source file may change a
   lot while preserving the same learned mechanism; a mechanism may change
   while remaining in the same broad family; trajectories may depart from their
   own histories while varying more or less across the population. The
   apparent conclusion depends on which level and representation are measured.
3. The tasks create a useful escalation ladder. Addition has a hard
   qualification boundary, so an audacious edit often becomes a complete
   failure even if it nearly works. Fashion-MNIST has a fixed-exposure,
   fixed-parameter budget and a noisy integer-correct primary objective, so
   rare improvements can be large but brittle. NanoGPT has a fixed time budget,
   so architecture changes can alter both learning quality and the amount of
   data processed. A search intervention must negotiate different feasibility
   landscapes rather than one generic benchmark.

Possible stronger opening sentence: “A long-running research agent is partly a
memory of its own successes.” The source, selected parent, and evidence ledger
make prior choices materially easier to extend than alternatives. The paper's
intervention operates against that endogenous asymmetry.

## 2. Related work

### 2.1 Measuring creativity in research agents: additional material

1. Bhushan et al.'s distinction among within-run novelty, historical novelty,
   and usefulness is particularly important here. The present lexical and
   source distances are within-run departure measures; the paper never treats
   them as proof of historical discovery. The executable evaluator and
   descendant record instead supply separate usefulness evidence.
2. Tang and Yang show concentration around seed literature at population
   scale. The present study complements that result by manipulating a direction
   during ongoing work and finding stable local departure; its own
   population-level conclusion is construct-dependent.
3. Ning et al. provide the closest systems precedent: auditable hypotheses,
   edits, evaluator results, and failure labels. This paper uses that trace
   granularity to study an intervention rather than merely to document a strong
   research system.
4. Heuresis is the closest search-architecture study: it compares greedy,
   archive, evolutionary, and divergent strategies inside an autonomous ML
   loop. The distinction here is the repeated semantic checkpoint and local
   executable transition, not invention of search-policy diversity.
5. IDEAgent is the closest lineage-management precedent. It treats research
   idea generation as quality-diversity search; the present paper instead
   audits exact ancestry as an outcome under a much simpler four-lineage
   moderator.

### 2.2 Eliciting alternatives and challenging assumptions: additional material

1. Denial prompting is the closest prompt-level comparator because it forces
   code alternatives by progressively excluding already used methods. The
   assumption challenge differs in leaving the alternative unspecified; that
   freedom is precisely why population diversity must be measured rather than
   assumed.
2. FirstResearch makes assumptions, mechanisms, falsifiers, and update rules
   explicit at research-question formation. The current study instead inserts
   assumption analysis after a trajectory has accumulated a selective history,
   so it can observe departure from an endogenous incumbent narrative.
3. Self-reflection and self-correction are not equivalent controls. Their usual
   target is error repair in a current answer. Here the ordinary agent may be
   entirely correct; the intervention asks it to redirect the scientific search
   operator, and an external evaluator determines whether the result works.
4. Sustained-creativity decoding studies long search quests and warns that one
   uniform creativity mode can still yield homogeneous alternatives. It is a
   direct precedent for the paper's construct-sensitivity check.
5. HypoSearch explicitly explores bounded hypotheses before commitment in deep
   research. It strengthens the case for branch-aware evaluation while leaving
   executable ML patches and periodic post-commitment challenges open.

### 2.3 Memory, lineages, and population diversity: additional material

1. The self-history literature provides at least three nonexclusive mechanisms:
   ownership-specific persistence toward one's own answer; autoregressive
   continuation from earlier explanations; and retrieval of similar episodes
   that induces similar outputs. This experiment cannot separate them because
   each prompt includes current code and a curated recent-results ledger but no
   continuous model conversation.
2. Greedy retention can be rational and still create a scientific blind spot.
   Strict improvement protects objective quality, but it removes executable
   counterexamples and near misses from parent selection. The evidence ledger
   remembers their summaries, not their source as a viable branch. This makes
   the evaluator/selector part of fixation, not merely a neutral judge.
3. Human design-fixation research warns that examples can improve quality while
   reducing variety. That is a useful counterargument to the assumption that
   diversity is always good: a locally narrow trajectory may be efficient when
   the incumbent family contains many exact redundancies, as in the addition
   controls.

4. Generative-AI design studies report fixation displacement: people can leave
   an initial example but converge on features suggested by the AI. This is a
   useful warning, not a result established by the present population metrics.
5. Multi-agent diversity-collapse work attributes convergence partly to
   interaction topology. These runs do not communicate, and their population
   direction reverses across semantic representations, so no comparable
   convergence mechanism should be inferred.

1. Formal prompt compliance is easy to observe but weak. Assumption language
   rises by 77.5–87.5 percentage points at challenged checkpoints, partly
   because the prompt explicitly asks for it. The harder evidence is the
   executable source change and whether later proposals inherit the mechanism.
2. Three failures have no final agent message because the provider returned no
   usable proposal. They remain in validity/retention outcomes. This illustrates
   why a text-only corpus would subtly select on model success and make the
   intervention look better than the actual research system.
3. Fresh initialization on every evaluation is scientifically important. It
   prevents a candidate from inheriting favorable training state, but it also
   means a conceptually exact parameterization can fail because it changes
   optimizer geometry or random-stream consumption. Addition messages learn to
   discuss quotient-aware initialization, clipping, and AdamW dynamics; those
   are real parts of whether a scientific idea is executable.

The contribution is not any one of assumption prompting, code search, novelty
measurement, or population memory. It is the longitudinal combination: insert
an open-ended challenge into executable research; measure the local message,
artifact, feasibility, cost, and objective transition; follow ordinary
follow-up windows and exact descendants separately; and compare individual
departure with construct-sensitive population dispersion. A
standalone novelty-positioning section would overstate this boundary, so the
paper communicates it through concrete comparisons in the introduction,
related work, design, and discussion.

### Research questions: additional hypotheses

1. Before looking at results, one could plausibly predict that four-lineage
   memory reduces the marginal value of challenge because alternatives already
   survive. The observed interaction is not stable across tasks, so the paper
   properly rejects that simple hypothesis.
2. One could also predict that a challenge always increases code size. That
   fails on addition because ordinary gauge edits are already code-heavy; the
   challenge shifts conceptual content without reliably increasing AST or line
   distance.
3. A useful follow-on RQ is whether a challenge should be scheduled or
   triggered. The 200-proposal Fashion-MNIST traces repeatedly revisit spatial
   ideas late in the run, suggesting that calendar-time prompting can itself
   become routine.

## 3. Study design

### 3.1 Three research environments: additional details

1. Addition starts from a 1,644-learned-parameter transformer at 99.96% public
   accuracy. The objective is not “make addition work” but preserve at least
   99% exact accuracy while shrinking the learned model. A one-parameter gain is
   valid scientific progress under this objective, which explains why ordinary
   trajectories can rationally exploit exact symmetries for many proposals.
2. Fashion-MNIST's validation score encodes correct-count first and cross
   entropy only as a tie-break. For reader intuition, a +31 score change is
   normally 31 additional correct images, whereas a fractional-only change
   usually improves calibration at the same correct count. This distinction
   helps interpret rare large challenged gains versus many small control gains.
3. NanoGPT fixes training time rather than tokens or steps. A slower mechanism
   may learn better per token yet see less data; the evaluator records total
   tokens, steps, MFU, VRAM, depth, and parameter count so those trade-offs are
   visible. The positive +581-second local evaluator-time difference includes
   queue/runtime effects and very slow candidates; it should not be equated
   directly with the 300-second training target.

The 6,080 proposals break down as 1,600 addition, 4,000 Fashion-MNIST, and 480
nanoGPT. The 304 treated checkpoints break down as 80, 200, and 24; matched
ordinary checkpoints double the message-audit denominator to 608.

### 3.2 Matched trajectories and intervention: additional details

1. All four conditions in a block share the same run seed, but model sampling
   is not deterministic enough to yield identical early proposals. Across
   matched challenge/control pairs, proposal-9 incumbents are almost always
   different. Shared seed is blocking, not exact prefix matching.
2. The four-lineage parent policy first fills open lineages, then chooses the
   least-selected lineage with score/age/id tie-breaks. This provides bounded
   architectural memory but not unrestricted population evolution. Challenge
   and ordinary pairs in that stratum can therefore see different selected
   parents at the same proposal even within a shared block.
3. The challenge has several components bundled together: step back, identify
   assumptions, seek a genuinely different learned mechanism, avoid repeating
   failed types without new evidence, and state old/new approaches. This paper
   estimates the package, not which sentence causes the effect.

### 3.3 Controlled interface and evidence: additional details

1. The response format asks for a falsifiable hypothesis and evidence before
   exact patch blocks. Because the same format is used in controls, extra output
   at challenged checkpoints is attributable to the direction's demands and
   larger implementation, not to a different schema.
2. Each proposal is a fresh model call, so there is no hidden provider-side
   conversation memory. Longitudinal dependence is carried through visible
   code, selected parents, and the recent-results ledger. This narrows the
   “memory” construct to scaffold-managed scientific state.
3. The evidence ledger includes unsuccessful work when a useful subject-level
   reason exists. That design lets later messages distinguish “the family is
   wrong” from “the implementation was malformed,” but summaries may also
   canonize an incorrect explanation. The qualitative audit finds examples of
   both productive diagnostic use and repeated rationalization.

## 4. Measurement and identification

### 4.1 Process outcomes: additional details

1. Normalized Python-token 3-grams replace identifiers, numbers, and strings
   with generic tokens before comparison. This reduces reward for renaming or
   constant changes, but it can also understate important hyperparameter edits.
2. AST-node multiset distance detects changes in structural composition but
   discards ordering and semantics. It is useful only in triangulation with
   token distance, changed lines, message mechanism, and evaluator outcome.
3. “New family” means new relative to that trajectory's prior recorded
   proposals, not new to ML, the task, or humanity. The paper deliberately uses
   *within-run novelty* or *departure* instead of historical creativity.

### 4.2 Estimand: additional details

1. The local difference-in-differences removes stable level differences and a
   simultaneous control transition, but it does not fully balance incumbent
   architecture, evidence history, or model stochasticity. It is best read as
   a repeated matched interruption contrast.
2. Bootstrap resampling uses blocks, not proposals. This makes intervals wide
   on nanoGPT because it has only three blocks, even though there are 24
   checkpoints. Treating each checkpoint as independent would create false
   precision.
3. Repeated intervention checkpoints are not independent “doses.” A successful
   challenge at proposal 10 changes the parent and evidence available at
   proposal 20. Later estimates describe the continuing policy of periodic
   challenge, not repeated resets to a common counterfactual state.

### 4.3 Qualitative audit: additional details

1. The deterministic sample prevents selecting only attractive anecdotes: each
   task-by-memory stratum contributes its largest source departure, largest
   retained gain, and first invalid alternative where available.
2. Complete patch inspection matters because a mechanism summary can exaggerate
   novelty. The three anchor patches—differential image basis, token bottleneck,
   and query-conditioned attention gates—do implement the advertised new
   computation.
3. A stronger future audit would use two blinded human coders on deidentified
   messages, predeclare the codebook, adjudicate disagreements, and separately
   rate novelty, evidence quality, and mechanism fidelity.

### 4.4 Integrity and denominator checks: additional details

1. Missing messages are one challenged addition provider failure, one ordinary
   Fashion-MNIST provider failure, and one challenged Fashion-MNIST provider
   failure. They are not silently imputed as noncompliance.
2. Source availability is 95.4% across all proposals because malformed/no-edit
   proposals have no meaningful candidate-parent comparison. Feasibility
   outcomes retain the full denominator, while source metrics use available
   pairs.
3. Prompt manifests and hashes support treatment verification. The analysis
   additionally searches prompt text for the distinctive “step back from the
   current line of work” direction and confirms its placement matches condition
   and schedule.

## 5. Immediate effects: redirection has a price

### 5.1 Public proposals leave the local narrative: additional insights

1. Addition shows the largest lexical-novelty jump (+0.193) even though its code
   structure does not clearly jump. The intervention changes conceptual framing
   from local gauge pruning to routing/position/token representations, but both
   can require similarly elaborate Python.
2. Causal-evidence language increases by +0.225 on addition and +0.330 on
   Fashion-MNIST, but is not higher on nanoGPT after local adjustment. This
   warns against assuming the prompt universally improves scientific
   explanation quality.
3. Numerical evidence is already near universal in both arms (95.5–100% at
   checkpoints), so the challenge does not create evidence citation from
   nothing. It changes how evidence is used to justify a departure.

### 5.2 Executable source changes: additional insights

1. Fashion-MNIST challenged patches average 41.6 changed lines at checkpoints
   versus 8.94 for controls; source novelty is 0.0640 versus 0.0298. This is a
   large practical difference in implementation scope.
2. NanoGPT controls are extremely local at checkpoints (AST distance 0.00149),
   while challenges reach 0.01987. Even though both numbers look small, the
   roughly 13× ratio reflects controls that often tune one scalar/window and
   challenges that add new learned pathways.
3. Addition controls average more changed lines than challenges because exact
   quotient implementations must preserve initialization and optimizer
   semantics. “Larger patch” is therefore not a task-invariant proxy for more
   conceptual novelty.

### 5.3 Novel proposals fail more often and cost more words: additional insights

1. Fashion-MNIST validity falls from 62% at control checkpoints to 31% at
   challenged checkpoints, and retention from 36.5% to 5%. The difference
   between validity and retention shows that some alternatives run but do not
   improve; both engineering complexity and objective risk contribute.
2. Addition challenged checkpoints output about 7,052 tokens versus 4,118 for
   controls. Unlike Fashion-MNIST and nanoGPT, complete message word count does
   not rise clearly, suggesting the output-token cost is carried by intricate
   patches/reasoning rather than longer public summaries.
3. The pre-intervention placebo token differences are −274, −75, and −173,
   whereas challenged checkpoints add roughly +3,380, +2,765, and +4,765
   output tokens. That contrast makes the cost effect especially difficult to
   explain as a stable condition-label imbalance.

## 6. What the messages and descendants reveal

### 6.1 Addition: extra cases

1. In block 1, single-incumbent proposal 10 replaces full positional vectors
   with a seven-dimensional mean-free quotient and removes 23 parameters while
   qualifying. The preceding ordinary proposal removes one local normalization
   bias. This is a smaller but exceptionally clean demonstration that a
   challenge can turn exact invariance into a whole-interface change.
2. A later full key/value-sharing proposal fails at 36.38% accuracy. The next
   challenge does not blindly repeat it: it shares one value channel while
   preserving private keys and most private values. That proposal still fails,
   but its rationale isolates which assumption remains untested.
3. In a four-lineage run, learned relative-distance attention removes roughly
   115 parameters and qualifies after query/key sharing fails. Portfolio memory
   exposes several designs, allowing the message to preserve independent
   projections while relocating position into attention.

### 6.2 Fashion-MNIST: extra cases

1. Block 1 four-lineage proposal 10 replaces a sequential, location-specific
   head with parallel standard/dilated features and global statistics. It is
   retained with a +90 score gain—the largest sampled success—and directly
   states that ensemble refinements had plateaued.
2. Block 3 single-incumbent proposal 60 adds identity-initialized global channel
   attention and is retained, moving from 9,133 to 9,162 correct. Later
   challenged variants repeatedly refine descriptors but fail, showing how one
   successful challenge can become a new attractor.
3. At proposal 120, a challenged class-specific log-mean-exp branch is valid but
   not retained. This is scientifically informative: preserving the flattened
   head addresses prior global-pooling failures, yet the alternative still does
   not outperform the incumbent.

### 6.3 Language modeling: extra cases

1. Block 1 proposal 10 adds role-specific bigram embeddings after all-local
   attention processes more tokens but worsens bits per byte. The candidate is
   invalid, showing that a sensible different context representation can fail
   at implementation/verification.
2. Block 2 proposal 10 adds a low-rank lexical residual readout and is retained,
   albeit with only a 0.000047 bpb gain. It redirects away from grouped-query
   throughput tuning toward value-path expressivity.
3. Block 1 proposals 20–40 successively test a low-rank bigram bypass,
   head-wise global context, and parameter-matched SwiGLU. None is retained,
   but their evidence tracks an emerging conclusion: limited global computation
   is load-bearing, and static context-window tuning has diminishing returns.

## 7. Follow-up search, measurement sensitivity, and memory

### 7.1 Policy-window versus exact-lineage value: additional insights

1. On addition, the challenged checkpoint contributes about 65% of the
   challenged-minus-control ten-proposal gain (30.05 of 46.19 parameters); the
   rest occurs during ordinary follow-up proposals. Those proposals are not all
   descendants of the checkpoint candidate.
2. On Fashion-MNIST, immediate raw gain is only 0.80 of the 3.70 cycle
   difference. The cycle effect remains positive in the original blocks
   (+4.07) and added blocks (+3.15), even though the local immediate-gain DiD is
   sensitive.
3. On nanoGPT, immediate challenged gain is worse by 0.000120 bpb, while the
   nine follow-ups gain an extra 0.001288, for a net +0.001168 cycle. This is the
   clearest reason to report windows, but not to equate them with lineage value.
4. Recursive parent tracing finds challenged/ordinary cycles with any exact
   descendants in 25/31 of 80 addition cycles, 10/69 of 200 Fashion-MNIST
   cycles, and 5/8 of 24 nanoGPT cycles. Lower challenge retention mechanically
   reduces downstream branch opportunities.
5. Descendant-only challenged-minus-control gain is +7.40 parameters on
   addition, −0.435 Fashion-MNIST score, and +0.000198 bpb reduction on
   nanoGPT. Anchor-plus-descendant branch gain is positive on all three, but
   those descriptive means do not identify mediation.

### 7.2 Population measurement sensitivity: additional insights

1. Full-rationale family distance is lower for challenge in all tasks (.427 vs
   .494 addition; .532 vs .628 Fashion-MNIST; .403 vs .581 nanoGPT). This view
   includes old assumptions, failed alternatives, and evidence—not only the
   candidate mechanism.
2. Applying the same regex taxonomy only to the dedicated mechanism field
   reverses every result: .813 vs .622, .774 vs .672, and .815 vs .704.
3. Assigning the first-mentioned primary family also shows greater challenged
   dispersion: .875 vs .828, .812 vs .681, and .833 vs .717. This is not a
   minor sensitivity; it invalidates a collective-convergence headline.
4. The disagreement suggests a new linguistic observation: challenged agents
   may cite the same task constraints and rejected families while proposing
   different mechanisms. Full-rationale similarity and mechanism similarity
   answer different questions.

### 7.3 Endpoint interpretation: additional insights

1. Addition is the strongest long-run case: challenged trajectories start only
   20 parameters better on average but end 389.5 parameters better, and all ten
   paired progress differences favor challenge.
2. Fashion-MNIST is mostly catch-up: challenged runs start 71.1 score units
   behind, improve more, and end only 2.9 units ahead on average with mixed
   block-level final differences. It supports recovery, not decisive endpoint
   dominance.
3. NanoGPT is pure catch-up at this horizon: challenged runs start 0.006282 bpb
   worse, close most of the gap, but remain 0.001610 bpb worse. A paper focused
   on final scores alone would miss the process change and possibly conclude
   the intervention failed.

### 7.4 Portfolio memory: additional insights

1. Addition's validity/retention penalty is concentrated in single-incumbent
   runs; four-lineage runs show roughly no local penalty. On Fashion-MNIST,
   retention penalty is larger with four lineages. On nanoGPT, the retention
   penalty is single-incumbent only. There is no task-general sign.
2. Four-lineage memory may change both diversity and difficulty: selected
   parents can be weaker or older, and the agent sees a reference design. That
   means “more memory” is not a one-dimensional increase in context.
3. With only 5/5/3 blocks by task, interaction estimates are especially
   unstable. A dedicated memory paper should predeclare memory as primary and
   increase blocks rather than reuse these moderator contrasts.

## 8. Implications for AI-supported science

### 8.1 Assumption challenge as search operator: additional ideas

1. The prompt has an exploration cost analogous to mutation-strength control:
   stronger departures create more invalid candidates and larger token bills.
   An adaptive controller could tune challenge strength to evaluator slack or
   stagnation length.
2. Failed challenged proposals can still improve scientific knowledge if the
   failure isolates an assumption. The current strict selector records the
   summary but gives no formal reward for information gain; a future evaluator
   could separately score discriminating tests without allowing them to replace
   the objective incumbent.
3. Event windows and lineage subtrees should both be reported. A window captures
   the policy regime; recursive ancestry captures development of the prompted
   artifact. Neither can substitute for the other.

### 8.2 Diversity and construct sensitivity: additional ideas

1. A useful dashboard would plot each proposal on two axes: distance from its
   own trajectory history and distance from contemporaneous peers, with a
   toggle among full rationale, mechanism field, source, and human coding.
2. Randomized paraphrases may reveal whether population diversity is caused by
   exact wording or task affordances. A conclusion should survive multiple
   predeclared semantic representations before being named.
3. Quality-diversity archives could retain one qualified representative per
   mechanism niche, reducing the need to repeatedly command novelty in the
   prompt.

### 8.3 Design recommendations: additional implementation detail

1. A stagnation trigger could be predeclared as no strict improvement for five
   proposals plus at least three same-family attempts. This ties the challenge
   to evidence while avoiding subjective operator intervention.
2. “Generate alternatives before selection” should be bounded and logged: ask
   for three concise assumption/mechanism pairs, select one using a separate
   criterion, and only then request a patch. Otherwise hidden alternatives make
   token accounting and causal interpretation impossible.
3. Narrative reset and artifact retention can be crossed experimentally:
   preserve incumbent code and raw evaluator results while removing the agent's
   prior explanations. This tests whether fixation is carried by the artifact,
   the result ledger, or self-authored prose.

### 8.4 Limitations: additional detail

1. The block seeds are shared across conditions, but provider sampling and
   asynchronous execution can differ. There is no provider-side attestation
   that every request used identical hidden sampling state.
2. Fashion-MNIST's public validation split is repeatedly optimized, so later
   scores may overfit the search set. The paper uses those values to analyze
   process, not claim generalization; final holdout evaluation would be needed
   for scientific model comparison.
3. The intervention repeats every ten proposals regardless of whether the
   previous alternative is still being exploited. It may interrupt productive
   local development as well as escape stagnation.
4. Mechanism families are multi-label regex matches. Full-rationale coding
   includes evidence and rejected approaches; mechanism-only and primary-family
   codings reverse its direction. Human blinded coding remains preferable.
5. Ordinary checkpoints are not length- or deliberation-matched. The observed
   package combines assumption diagnosis, novelty pressure, contrastive
   explanation, and extra reasoning.
6. One fresh training evaluation determines retention. Repeated seeds would be
   needed to quantify threshold stability, especially for fixed-time nanoGPT.

## 9. Conclusion: additional synthesis

1. The intervention is simultaneously successful and inefficient: it changes
   the search distribution in a verifiable way but spends more tokens and
   produces fewer immediately retained candidates. Any binary “works/doesn't
   work” conclusion discards the central result.
2. The artifact, not the agent's claimed realization, is the potential carrier
   of redirection. Exact ancestry confirms that token bottleneck, differential
   input basis, and attention-gate examples can be refined, while showing that
   such descendant branches are uncommon overall.
3. Population claims are not guaranteed by one rationale-level taxonomy. A
   robust study must preserve independent evidence paths and audit conclusions
   across mechanism text, source behavior, and blinded human coding.

## Appendix A: extra prompt/protocol notes

- The exact challenge is a package treatment; future ablations should separate
  “step back,” explicit old/new assumption reporting, prohibition on repeating
  failures, and the mechanism-level requirement.
- The public metadata format itself may regularize scientific behavior. Because
  it is constant across arms, it does not explain the contrast, but results may
  not transfer to free-form coding agents.
- Calling the system “greedy OpenEvolve-style” is important. Full OpenEvolve
  adds population database, islands, inspirations, and diversity mechanisms;
  findings should not be attributed to that whole official architecture.

## Appendix B: extra quantitative options

- Report raw checkpoint means alongside DiD: Fashion-MNIST source novelty is
  0.0640 challenge versus 0.0298 control; nanoGPT 0.01186 versus 0.00290;
  addition 0.02080 versus 0.01680.
- Report prompt compliance as a denominator-aware manipulation check:
  assumption language is present in 87.5%, 78.0%, and 87.5% of challenged
  checkpoint records for addition, Fashion-MNIST, and nanoGPT, versus 2.5%,
  0.5%, and 0% in controls.
- Report costs in absolute checkpoint means: treated output tokens average
  7,052 addition, 4,926 Fashion-MNIST, and 6,317 nanoGPT, versus 4,118, 1,665,
  and 1,676 in controls.

## Appendix C: extra qualitative cases

- Addition block 5 four-lineage proposal 10 implements a positional
  gauge-quotient while preserving AdamW-equivalent updates and qualifies,
  directly addressing a prior 4.5% failure attributed to optimizer/RNG
  disruption.
- Fashion-MNIST block 3 four-lineage proposal 20 rewrites the classifier around
  class-conditional multiscale evidence pooling; it is the largest sampled
  source departure (distance 0.635) and fails. It exemplifies high novelty with
  low feasibility.
- NanoGPT block 1 single-incumbent proposal 30 reallocates global context from
  occasional full layers to one full-context head per layer. It is valid but not
  retained, and its evidence explicitly brackets two versus three full-context
  layer results.

## Appendix D: extra reproducibility notes

- The analysis has no network calls, model calls, or evaluator execution. It
  only reads frozen records and candidate source snapshots.
- The bootstrap seed is fixed at 20260903. Numerical CSV/JSON outputs are
  deterministic; figure image bytes can vary with renderer metadata.
- A release artifact should add a raw-input hash ledger and either package the
  relevant run slices or provide a script that validates externally supplied
  campaign directories before analysis.
