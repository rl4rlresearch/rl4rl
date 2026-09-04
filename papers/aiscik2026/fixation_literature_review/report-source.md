# Fixation in LLM Research Agents

## A cross-disciplinary evidence map for trajectory lock-in, defixation, memory, and search diversity

**Research synthesis prepared 3 September 2026**
**Scope:** autonomous and iterative LLM systems, with special attention to the state-matched model-compression trajectories in this repository.

## Executive answer

The fixation observed in the project is real enough to have several strong empirical relatives, but the relevant research is fragmented across fields and names. The closest terms are **design fixation**, **anchoring**, **choice-supportive or self-preference bias**, **history contamination**, **experience-following**, **premature convergence**, **diversity collapse**, **generative monoculture**, and **exploration-to-exploitation collapse**. No single one of those constructs is identical to the project's outcome.

The most defensible name for the project construct is **trajectory fixation**: a persistent concentration of an agent's executable proposals around an incumbent mechanism or local edit family, relative to alternatives that remain available, with persistence produced by the interaction of model, prompt history, memory, evaluator, and selection policy. This is a behavioral description, not a claim about private mental states.

Five conclusions survive the literature review:

1. **Prior answers and examples can become causal attractors.** LLMs are affected by red herrings and numerical anchors; seeing their own prior answer can make them less likely to change it and more confident in it; and self-refinement can amplify preference for self-generated content [S01-S05]. Human design-fixation studies provide the older analogue: examples narrow the explored space and induce feature copying, including when the starting idea was self-generated [S18-S24].
2. **Longitudinal history can propagate the attractor.** Multi-turn models prematurely commit, then rely on their own earlier attempts; retrieved agent memories elicit similar outputs and propagate errors; continuously consolidated memories can eventually harm performance; and omitting assistant history or using summaries can sometimes outperform full transcripts [S06-S10].
3. **The scaffold is part of the cause.** Greedy selection, dense inter-agent communication, structured prompt templates, shared exemplars, and naïve memory growth all reduce effective exploration in at least some settings [S08, S11-S17, S29-S33]. Fixation is therefore not simply a fixed property of the base model.
4. **Breaking one fixation can create another.** Human-AI design work calls this *fixation displacement*: a user escapes the initial example but converges on the AI's suggestion. The repository shows an unusually clean machine-only version: the challenge prompt displaced width/bias micro-pruning, but 17 of 32 treated forks independently converged on token-interface factorization [S15, P2]. This is a shift of attractor, not unconstrained diversity.
5. **Novelty and usefulness must be measured separately.** Examples can reduce variety while improving idea quality; assumption challenges can increase structural novelty while lowering qualification; and recent ML-agent work finds declining within-run novelty and weak coupling between novelty and performance [S19, S27, S34-S38]. A defixation intervention succeeds only if its different ideas are executable, informative, or open productive descendants.

The project therefore sits at a valuable intersection. The surrounding literature establishes pieces of the phenomenon, while the project's exact shared-prefix forks test a gap: whether one semantic prompt, inserted into the same executable research state, changes the next mechanism, feasibility, cost, and downstream search path.

## 1. What “fixation” should mean here

### 1.1 Operational definition

For an autonomous research trajectory, fixation is not merely “repetition,” low temperature, or low textual diversity. A useful operational definition has four parts:

- **An incumbent attractor:** a mechanism, representation, edit family, hypothesis, parent lineage, or problem framing that recurs.
- **Available alternatives:** the task permits materially different executable approaches; otherwise persistence may be rational convergence.
- **Excess persistence:** reuse exceeds a relevant counterfactual—another prompt, a state-matched fork, a reset, a different memory policy, or a diverse population.
- **Search consequence:** persistence affects mechanism coverage, feasibility, improvement, cost, recovery, or branch survival.

This definition deliberately separates observable behavior from anthropomorphic claims. The agent need not “believe” the incumbent. Autoregressive continuation, prompt imitation, retrieval, and deterministic selection can jointly create the same behavior.

### 1.2 Seven layers that are often conflated

| Layer | What is repeated or narrowed? | Best diagnostic | Closest literature |
|---|---|---|---|
| Prior/model | High-probability patterns learned before the run | Across-seed population similarity | generative monoculture, alignment diversity |
| Prompt/exemplar | Features of a shown answer, hint, or template | Remove, randomize, abstract, or diversify examples | anchoring, red herrings, design fixation |
| Self-history | The agent's own plans, code, rationales, and failures | Hide assistant history; state-matched transcript ablation | self-bias, choice-supportive bias, lost-in-conversation |
| Memory/retrieval | Retrieved episodes or consolidated summaries | Episodic versus consolidated memory; selective deletion | experience-following, memory poisoning |
| Selection/evaluator | Locally successful edits or score-compatible behavior | Greedy versus population or quality-diversity policies | premature convergence, Goodhart-style adaptation |
| Social/population | Shared consensus across interacting agents | Independent versus dense communication; diversity by topology | structural coupling, false consensus |
| Representation | One mechanism family dominates executable artifacts | Source/AST/mechanism transition metrics | design fixation, P-creativity |

An experiment can show several layers at once. A long greedy trajectory with full history, a single incumbent, and a strict evaluator is an interacting system—not a clean assay of a base-model trait.

### 1.3 What fixation is not

- **Not ordinary exploitation.** Once an architecture produces a large gain, exploiting it can be rational. The empirical question is whether exploitation persists after evidence of diminishing returns or blocks valuable alternatives.
- **Not low lexical diversity.** Different wording can encode the same mechanism; nearly identical wording can conceal a large code change.
- **Not failure to obey a novelty request.** Prompt compliance is a manipulation check, not the outcome.
- **Not always harmful.** Concentration can improve consistency, feasibility, or depth. In the 43-study design-example meta-analysis, examples reduced variety but improved novelty and quality on average [S19].
- **Not model collapse.** Recursive training on synthetic data can erase distribution tails [S45], but that is a training-distribution phenomenon. It is only a distant analogy to within-run search lock-in.

## 2. The most direct LLM evidence

### 2.1 Red herrings and anchors

Naeini et al.'s *Large Language Models are Fixated by Red Herrings* uses the Only Connect Wall, where 16 words contain four intended groups and distracting associations. GPT-4 and GPT-3.5 were far below humans on the original walls, and performance rose dramatically when word order or WordNet-based construction diluted the red-herring structure [S01]. The clean lesson is that salient but irrelevant associations can dominate grouping. The WordNet version also changes difficulty, so its very large gain is not a pure causal estimate of red-herring removal.

Lou and Sun experimentally induce numerical/semantic anchors and report that biased hints shift LLM responses. Chain-of-thought, reflection, “ignore the anchor,” and principle-based prompts were insufficient; gathering hints from multiple angles worked better [S02]. This supports an important protocol lesson: a generic instruction to “think differently” may be weaker than changing the evidence set.

### 2.2 Self-generated answers are unusually sticky

Kumaran et al. provide the strongest controlled evidence for a self-history mechanism [S03]. In stateless two-stage queries, an answer was either visible or hidden before the model reconsidered it. For the main Gemma 3 12B experiment, the change-of-answer rate was 13.1% when the answer was shown versus 34.0% when hidden—a 71% reduction in the odds of changing. Visibility also raised confidence. The effect generalized across six tested models and additional factual and mathematical tasks. Crucially, telling the model that the displayed answer came from another LLM largely removed the ownership-specific effect. This makes “the model's own prior output” more than just another token sequence in this paradigm.

The same paper prevents a simplistic story. Models also overweighted opposing advice by roughly 1.5-3.4 times the Bayesian update across models. LLMs can therefore be sticky toward their own visible answer and overly pliable toward explicit contradiction. “Fixated” is context-dependent, not a permanent personality.

Xu et al. similarly find self-bias across translation, constrained generation, and mathematics: models tend to prefer their own generations, and iterative self-refinement can amplify the bias [S04]. Larger models and accurate external feedback mitigate it. Chen et al. later show why measuring self-preference needs a quality-controlled gold baseline: otherwise a model may favor its own answer because it is genuinely better [S05].

### 2.3 Long conversations create self-reinforcing assumptions

Laban et al. evaluate 15 models on more than 200,000 simulated conversations across six generation tasks [S06]. Sharding an otherwise complete instruction across turns produces an average 39% performance drop. The authors attribute the effect mainly to unreliability: models prematurely propose full answers, introduce unsupported assumptions, over-rely on earlier wrong attempts, overweight the first and last turns, and produce verbose histories that add further assumptions. More test-time reasoning did not remove the effect. Repeating user requirements helps: a final recap improves more than a cumulative “snowball,” but neither matches giving the full specification once; realistic repetition recovers about 15-20% of the degradation.

This maps closely to long research sessions. Every proposal summary, failed edit, speculative explanation, and locally successful mechanism becomes part of the next proposal's effective environment. More context can be more *causal material*, not simply more knowledge.

Huang et al. test real multi-turn conversations and find that omitting assistant messages often preserves response quality while cutting context sharply; in some cases full assistant history causes rehashing of errors, hallucinations, or style [S07]. This is a 2026 preprint, but it directly motivates transcript ablations. A compact state summary may preserve user constraints while removing the model's own narrative inertia.

### 2.4 Memory makes experience-following measurable

Xiong et al. identify **experience-following**: retrieved memories that resemble the current input tend to make agent outputs resemble the retrieved output [S08]. Naïvely adding all experiences harmed four agents in their reported settings—for example, RegAgent fell from 67.53 to 55.48 and AgentDriver from 40.11 to 32.32—whereas strict, selective addition and deletion improved average absolute performance by about ten percent over naïve growth. Memory is therefore a policy over future behavior, not an inert database.

The concern becomes stronger when memories are repeatedly rewritten. Zhang et al. report that consolidated memory first helps and then degrades below a no-memory baseline; even consolidation from ground-truth episodes caused a frontier model to fail 54% of ARC-AGI items it previously solved without memory [S09]. The paper is a recent preprint, so replication is needed, but its episodic-versus-consolidated distinction is highly relevant. Raw episodes preserve counterevidence; compressed summaries can canonize one causal story.

Strategic-forgetting work likewise treats deletion as functional rather than accidental. SF-AMS reports gains from removing redundant or irrelevant structured memories [S10]. The broad memory literature now increasingly treats retrieval, update, deletion, and forgetting as separate design choices—not a scalar “has memory/does not have memory.”

### 2.5 Reflection can stall or echo

Li et al. name **Early Stop Reflection**: useful reflections concentrate early, after which repeated reflection produces diminishing novelty [S11]. Their DORA system generates task-adaptive diverse advice instead of repeating a fixed reflection prompt. Du et al. observe an **Echo Trap** in multi-turn reflection trained with GRPO: later reflections copy earlier ones because repetition preserves reward; a tree-structured turn-credit method restores exploration [S12]. The training setting differs from the repository's inference-only agents, but the dynamic—reward-compatible repetition crowding out fresh diagnosis—is analogous.

The larger self-correction literature urges caution. Reflexion shows that verbal feedback and episodic memory can improve agent performance [S39], but Kamoi et al.'s critical survey finds little evidence that ungrounded prompted self-feedback reliably causes correction outside unusually suitable tasks; reliable external feedback is much more dependable [S40]. Huang et al. likewise report that intrinsic self-correction can degrade reasoning without external signals [S41]. A defixation prompt should therefore be evaluated by code and evaluator outcomes, not by the persuasiveness of its self-critique.

## 3. Autonomous research and optimization agents

### 3.1 Direct evidence of narrowing scientific search

Tang and Yang evaluate four research-agent frameworks and six models, generating 37,802 ideas from shared seed literature [S13]. Their preprint reports that AI ideas are more concentrated than human-authored papers in the same areas, remain closer to seed papers than later human work, and tend to differ through recombination of established methods rather than new research questions. This is among the closest field-level analogues to trajectory fixation, although it studies idea text rather than executed architectural descendants and is not yet peer reviewed.

Bhushan, Zhang, and Wang's August 2026 COLM paper is the closest measurement paper to the current project [S14]. It segments AIDE and AIRA-Dojo trajectories on ten MLE-Bench competitions into executable episodes and separates P-creativity (new relative to the agent's own run), H-creativity (new relative to a large human corpus), feasibility, and impact. Across frameworks and models, P-creativity declines as iteration proceeds while performance rises; novelty is not reliably associated with impact. GPT-5 LLM-judge P-creativity correlates 0.732 with 300 human annotations, but requires the current episode plus prior episodes and remains an automated proxy. The study examines up to ten episodes in eight-hour runs; the repository's 70- to 200-proposal trajectories cover a much longer horizon.

MLAgentBench established iterative ML experimentation as an agent benchmark and found the strongest tested configuration succeeded on only 37.5% of tasks, with long-term planning and hallucination as recurring failures [S35]. AIRA-Dojo/MLE-bench work later shows that the interaction between search policy and operator set matters: changing the available operations and planning policy raises medal-level success from 39.6% to 47.7% in one study [S36]. This again locates behavior in the research system, not only the base model.

### 3.2 Premature convergence in LLM optimization

Carbonati et al. study LLM-driven Bayesian optimization and find a coupled “choose the strategy and generate the point” prompt unstable: depending on prompt order, the system oscillates between almost uniform exploration and greedy premature convergence. Decomposing policy selection from candidate generation improves acquisition behavior [S17]. The study is small and uses a different domain, but it offers a concrete mechanism: asking one model turn to decide both *how* to search and *what* to try can make strategy commentary collapse into one repeated operator.

The analogy to evolutionary computation is technically useful when kept precise. Premature convergence occurs when selection removes useful behavioral variation before the global search is complete [S42]. Novelty Search removes the objective gradient and selects for behavioral novelty, allowing escape from deceptive objectives [S43]. MAP-Elites and quality-diversity methods retain high-performing representatives across behavioral niches [S44]. These methods suggest archive-level interventions—retain qualified mechanisms across niches—rather than repeatedly commanding one greedy trajectory to be novel.

AlphaEvolve and OpenEvolve-style systems add population memory, evaluator feedback, islands, and inspirations to code evolution [S37]. They can preserve alternatives, but population machinery alone is not a guarantee. If prompt, parent selection, descriptor choice, and evaluator all reward the same local family, the population can contain many textual variants of one mechanism.

## 4. Population and interaction effects

### 4.1 Generative monoculture

Wu, Black, and Chandrasekaran find that aligned LLM output distributions can be substantially narrower than their source distributions in reviews and code, calling this **generative monoculture** [S29]. The narrowing can coexist with higher correctness or efficiency, and simple temperature or top-p changes do not reliably restore the lost diversity. Murthy, Ullman, and Hu similarly report that aligned models do not reach human conceptual diversity and are generally less diverse than instruction-tuned variants [S30].

Wenger and Kenett find a population-level version in standard creativity tasks [S31]. Across 102 humans and 22 LLMs, individual LLM originality can match humans, while between-output variability is markedly smaller. For example, mean variability was 0.459 for LLMs versus 0.699 for humans on an Alternative Uses Task. Higher temperature increases diversity only up to a point; at extreme settings coherence deteriorates. Strong single proposals and a narrow population can coexist.

### 4.2 Multi-agent systems can amplify convergence

Chen et al. vary model strength, roles, group size, and communication topology in open-ended multi-agent ideation [S16]. Stronger aligned models show diminishing marginal diversity; authority-heavy groups suppress semantic diversity; larger groups have diminishing returns; and dense communication accelerates premature convergence. Their term **structural coupling** captures how interaction synchronizes trajectories.

This does not mean all collaboration is harmful. Ueda et al. find that larger cohorts, deeper dialogue, persona heterogeneity, and critic-side diversity can improve idea diversity or feasibility [S32]. The reconciliation is architectural: independent or heterogeneous generators plus selective criticism can broaden search, while dense consensus can collapse it. “Use more agents” is not an intervention unless topology, independence, and selection are specified.

### 4.3 Prompt format can itself be an attractor

Yun et al. show that role templates and special-token conversation formats can reduce semantic diversity even at high temperature [S33]. This matters for experimental prompts: requiring an exact response schema, naming a fixed list of mechanism families, or supplying canonical examples may reduce the very diversity a challenge prompt is intended to elicit. Format is a treatment component and should be held constant or separately ablated.

## 5. What human design fixation adds

### 5.1 The construct's origin

Jansson and Smith defined design fixation as blind adherence to limiting concepts and experimentally showed that designers copy features—including flawed ones—from example solutions [S18]. Later reviews emphasize that fixation is defined relative to the level of analysis: repeating a component may be fixation at one level and productive refinement at another [S20]. This is why the project should report line edits, AST changes, mechanism families, and lineage transitions rather than one scalar “diversity” score.

The 43-study meta-analysis by Sio, Kotovsky, and Cagan is the most important corrective [S19]. Examples cause narrower, deeper exploration, more example-related ideas, and fewer categories, yet can increase novelty and quality. A single uncommon example was especially beneficial. This predicts the project's novelty/feasibility trade-off: an attractor may be locally productive even while reducing population breadth.

### 5.2 Self-generated fixation and fixation displacement

Leahy et al. find that designers can fixate on their own initial concepts, not just externally provided examples; Design Heuristics introduced in the same session increased variation [S21]. That is a closer human analogue to long LLM trajectories than classic exemplar copying.

Wadinambiarachchi et al. randomly assign human designers to text-to-image assistance or search/no-AI conditions [S15]. AI assistance reduces idea quantity, variety, and originality and increases fixation on generated images. The authors describe **fixation displacement**: participants move away from an initial example only to copy the AI's output. In their corpus, 206 of 468 generated images (44%) depicted similar humanoid robots, and fixation in later sketches correlated with fixation in preceding AI images. This provides a ready conceptual model for a challenge prompt that reliably creates a new shared attractor.

### 5.3 Warnings are weaker than changing the task environment

Viswanathan and Linsey find that familiar examples create stronger fixation; warnings help more for unfamiliar than familiar features, while physical build-and-test exposes disadvantages and reduces fixation [S22]. Partial or incomplete visual stimuli can preserve inspiration while reducing copied surface features [S23]. Incubation and distributed effort can also help because uninterrupted work repeatedly retrieves the same concepts [S24-S26].

The transferable lesson is not to anthropomorphize incubation. It is to change retrieval and evidence: remove self-generated narrative, provide an abstract rather than concrete precedent, run a discriminating evaluator, or restart from the retained artifact with a different session seed.

## 6. Interventions: what works, what fails, and why

### 6.1 Evidence-ranked intervention table

| Intervention | Expected mechanism | Evidence | Main risk |
|---|---|---|---|
| Hide assistant history; retain task + artifact + results | removes self-anchoring and narrative contamination | controlled choice-supportive bias; assistant-history ablations [S03, S06, S07] | repeats failures if results are also lost |
| Selective episodic memory with deletion | prevents misleading experience-following | multi-agent memory experiments [S08-S10] | curator/selector introduces a new bias |
| External evaluator feedback | grounds revision in causal task evidence | self-correction survey; build-test design studies [S22, S40, S41] | evaluator gaming or narrow objective |
| Assumption challenge / denial prompting | prohibits or reframes incumbent mechanism | project fork; NEOCODER denial prompting [S38, P2] | novelty-feasibility loss; new shared attractor |
| Dynamic diverse advice | changes prompt conditional on stagnation/evidence | DORA [S11] | advice generator itself converges |
| Independent generators + delayed exchange | prevents structural coupling | multi-agent diversity work [S16, S32] | duplicated compute and weak synthesis |
| Quality-diversity archive | preserves high-quality behavioral niches | Novelty Search, MAP-Elites [S43, S44] | descriptors define what “different” means |
| Separate strategy choice from edit generation | prevents one operator from monopolizing both | LLM Bayesian optimization [S17] | coordination overhead |
| Recap user constraints, not assistant prose | restores task specification salience | lost-in-conversation [S06] | recap can omit evolving discoveries |
| Incubation / session reset with artifact retained | interrupts repeated retrieval while preserving progress | human incubation + transcript evidence [S24-S26] | loss of useful causal memory |
| Temperature increase | broadens sampling locally | mixed; monoculture persists; coherence drops at extremes [S29-S31] | invalid or low-quality proposals |
| Generic “reflect” or “ignore anchor” | prompts self-correction | generally weak without external evidence [S02, S40, S41] | confident rationalization |

### 6.2 Why the project's prompt worked at the fork

The intervention did more than request novelty. It explicitly redirected search from current assumptions toward different learned computational mechanisms, required a comparison between old and new assumptions, and occurred at a state where the ordinary trajectories were concentrated on width/bias pruning. That combination created a large, observable representation shift.

Across 32 exact state-matched Tiny Addition forks at opportunity 10 [P2]:

- normalized source novelty increased by 0.03085, AST distance by 0.03421, and changed lines by 13.13;
- a new broad mechanism tag appeared in 62.5% of treated proposals versus 9.4% of controls;
- qualification/retention fell from 50.0% to 40.6%;
- immediate parameter reduction increased by 536.8 parameters on average;
- output length increased by 2,451 tokens and evaluator time by 17.9 seconds.

The key mechanism evidence is categorical. Ordinary forks produced 18 feedforward-compression and 13 normalization/bias-pruning proposals. Treated forks produced 17 token-interface factorizations, nine attention-routing changes, four shared/iterative-depth designs, one cross-sublayer projection reuse, and one relative-position design. The prompt clearly changed the executable search distribution.

### 6.3 Punctuated defixation, not permanently elevated novelty

During opportunities 10-19, before the second challenge, treated trajectories ended 786 parameters smaller on average while qualification was only 3.1 percentage points lower. Mean source novelty across all ten proposals was only 0.00315 higher. The structural jump happened at the intervention; subsequent ordinary turns exploited the new mechanism. Through opportunity 70, treated trajectories ended 1,652 parameters smaller in 28 of 32 pairs, but had a 9.84-point lower qualification rate and no persistent source-novelty advantage.

This is best described as **punctuated defixation**: an intervention changes the basin, then ordinary exploitation resumes. It is stronger than a transient wording effect because the new artifact and its descendants persist. It is weaker than a claim that the agent becomes durably open-minded.

### 6.4 The displacement result is scientifically important

Fifty-three percent of treated forks independently chose token-interface factorization. That does not invalidate the intervention. It shows that a shared prompt plus shared baseline can define a new high-probability attractor. Future protocols should measure both:

- **departure:** distance from the incumbent family; and
- **dispersion:** pairwise distance among treated proposals and descendant lineages.

A treatment can score high on departure and low on dispersion. This distinction links the repository to fixation displacement [S15] and generative monoculture [S29-S31].

## 7. Measurement recommendations for the papers

### 7.1 Use a multi-level measurement battery

1. **Artifact-level departure:** normalized source-token distance, AST-node distance, changed lines, parameter topology, and execution trace.
2. **Mechanism-level transition:** predeclared or independently coded families; transition matrix; probability of switching away from the incumbent.
3. **Lineage-level persistence:** parent choice, branch survival, return to older branches, depth, and time since a lineage was last selected.
4. **Population-level dispersion:** pairwise mechanism distance, effective number of occupied niches, entropy, and clustering around a new attractor.
5. **Usefulness:** executable/qualified rate, immediate improvement, descendant improvement, regret, and discovery of valid counterevidence.
6. **Cost:** input/output/cached tokens, evaluator seconds, wall time, invalid attempts, and redundant descendants.

### 7.2 Metrics that directly diagnose fixation

- **Mechanism persistence:** longest run and proportion of consecutive proposals in the same family.
- **Switch hazard:** probability of a family change conditional on stagnation duration.
- **Time to first qualified alternative:** proposals/cost from an intervention to an executable mechanism outside the incumbent family.
- **Transition entropy:** entropy of outgoing family transitions, stratified by evaluator result.
- **Attractor concentration:** share of independent trajectories occupying the modal family at a checkpoint.
- **Fixation displacement index:** departure from control attractor minus dispersion within the treated population.
- **Qualified novelty:** novelty multiplied by qualification, plus a separate two-dimensional plot to avoid hiding the trade-off.
- **Descendant value:** future incumbent improvement attributable to a mechanism-changing proposal, not only its immediate score.
- **Stagnation regret:** best counterfactual branch score minus continued incumbent-line score, when a paired or archived counterfactual exists.

### 7.3 Identification hierarchy

Evidence should be labeled by design strength:

1. **Exact state-matched fork:** same artifact, prompt, history, evaluator state, and parent until insertion—strongest local estimate.
2. **Matched block:** same baseline and schedule but divergent prehistory—useful descriptive contrast.
3. **Within-run change point:** compare before/after intervention—confounded by time and stagnation.
4. **Cross-framework comparison:** different memory, parent selection, prompt, and evaluator interface—whole-system description, not a model-trait estimate.
5. **Text-only or endpoint correlation:** hypothesis-generating only.

### 7.4 Do not overclaim cognition

Recorded rationales can establish that the prompt changed the model's public proposal description; they cannot reveal private chain-of-thought. Phrase claims as “the trajectory switched mechanisms,” “the proposal cited an assumption,” or “the source departed from the parent,” not “the model realized” or “the agent believed.”

## 8. High-value next experiments

### 8.1 Factor history separately from the challenge prompt

At a matched checkpoint, fork each trajectory into:

- full transcript + ordinary prompt;
- full transcript + challenge;
- user/task constraints + compact evaluator-result ledger + incumbent artifact, but no assistant prose;
- same compact state + challenge;
- episodic raw failures retrieved by relevance;
- consolidated narrative memory.

This identifies whether the prompt works by adding a new instruction, by counteracting self-history, or by interacting with the incumbent artifact. It directly connects [S03, S06-S10] to the project's executable outcome.

### 8.2 Measure prompt-induced convergence

Randomize several semantically matched challenge prompts: abstract assumption challenge, denial of the incumbent family, request for multiple mechanisms before selection, and one uncommon cross-domain analogy. Keep response format constant. The primary outcomes should be departure, treated-population dispersion, qualification, and descendant value. This tests whether the current 17/32 factorization attractor is wording-specific.

### 8.3 Trigger by evidence, not calendar time

Compare fixed every-ten scheduling with a predeclared trigger such as: no incumbent improvement for five proposals, mechanism entropy below a threshold, or repeated evaluator failure with the same diagnosis. DORA and the exploration/exploitation literature predict that the value of defixation is state-dependent [S11, S14].

### 8.4 Compare greedy prompting with archive-level diversity

Cross prompt treatment with selection policy:

- one global incumbent;
- portfolio memory;
- MAP-Elites-style niches defined by learned mechanism/representation features;
- native population with islands;
- independent parallel trajectories with delayed evidence exchange.

If prompts only move one trajectory between attractors while an archive maintains several productive niches, this would distinguish semantic defixation from institutional pluralism.

### 8.5 Test periodic forgetting correctly

Retain the current best artifact and objective results, but reset assistant prose and generation seed. Compare this with a full-history continuation and a reset that also hides failed-result history. The former tests narrative fixation; the latter risks repeated failures. Outcomes should include duplicate-edit rate, mechanism coverage, qualification, and improvement per token.

### 8.6 External validity

Replicate on at least one task where “different mechanism” is not naturally synonymous with fewer parameters: Fashion-MNIST, nanoGPT, or an algorithmic task with a sealed correctness oracle. Use a different model family if available. Recent ML-agent work covers broad Kaggle tasks but short horizons [S14]; the repository covers long horizons but few task/model combinations. Together they suggest the right generalization program.

## 9. What is established, plausible, and still unknown

### Established with relatively direct evidence

- Visible self-generated answers can causally increase persistence and confidence [S03].
- Multi-turn interaction can produce early unsupported assumptions and reliance on prior wrong attempts [S06].
- Retrieved memories can make outputs imitate past outputs and propagate errors [S08].
- Examples and AI suggestions can narrow design variety; a new suggestion can become a replacement fixation [S15, S18-S23].
- Agent populations can converge because of communication topology and prompt format [S16, S29-S33].
- In this repository, the assumption challenge causally changes the immediate executable proposal distribution at exact shared-prefix forks [P2].

### Plausible synthesis, not yet directly proven

- The same ownership-specific mechanism in short answer revision explains long code-trajectory lock-in.
- Assistant prose is the main carrier of the repository's fixation rather than the incumbent code or evaluator landscape.
- The factorization convergence is caused by the exact wording of the challenge rather than a task-level architectural affordance.
- Portfolio memory reduces the marginal value of challenge prompts by already preserving alternatives.

### Open questions

- Can an agent learn when to explore rather than receive a fixed intervention schedule?
- What mechanism descriptors preserve genuinely different causal models rather than surface code variation?
- How should novelty credit propagate to descendants when the first proposal fails but introduces a useful representation?
- Can a memory system retain negative evidence without retaining the narrative framing that caused fixation?
- When is convergence epistemically appropriate rather than premature?
- Do state-matched effects transfer across base models, task domains, and evaluator regimes?

## 10. Bottom line for Paper 2

The literature strongly supports the existence of local lock-in mechanisms, but it also prevents a simplistic “LLMs are fixated” conclusion. LLMs can be anchored by external examples, their own prior outputs, summaries, memories, selection rules, and other agents; they can also be too eager to abandon correct answers under contradiction. Fixation is a property of a model-in-scaffold-in-history.

Paper 2's strongest contribution is therefore not a new cognitive bias claim. It is a **state-matched causal instrument for search behavior**. The intervention changed the next executable mechanism, produced a large immediate compression gain at a feasibility cost, displaced one population attractor with another, and created descendants whose advantages persisted after surface novelty returned to baseline. That “punctuated defixation” pattern is more precise, more defensible, and more interesting than saying the prompt made the LLM creative.

## Selected annotated bibliography

### Direct LLM fixation, anchoring, and self-history

- **[S01] Naeini et al. (2023), “Large Language Models are Fixated by Red Herrings.”** NeurIPS Datasets and Benchmarks. Direct named fixation benchmark; salient false associations dominate grouping. https://papers.nips.cc/paper/2023/file/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html
- **[S02] Lou & Sun (2024), “Anchoring Bias in Large Language Models: An Experimental Study.”** arXiv:2412.06593. Anchor induction and mitigation; simple reflection-style prompts are insufficient. https://arxiv.org/abs/2412.06593
- **[S03] Kumaran et al. (2026), “Competing Biases underlie Overconfidence and Underconfidence in LLMs.”** Nature Machine Intelligence 8:614-627. Controlled choice-supportive bias and contradictory-advice overweighting. https://www.nature.com/articles/s42256-026-01217-9
- **[S04] Xu et al. (2024), “Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement.”** ACL. Self-generated content is favored and can be amplified by refinement. https://aclanthology.org/2024.acl-long.826/
- **[S05] Chen et al. (2025), “Beyond the Surface: Measuring Self-Preference in LLM Judgments.”** EMNLP. Gold-adjusted measurement separates bias from genuine quality. https://aclanthology.org/2025.emnlp-main.86/
- **[S06] Laban et al. (2026), “LLMs Get Lost In Multi-Turn Conversation.”** ICLR. Large controlled study of multi-turn unreliability, premature answers, and reliance on prior attempts. https://proceedings.iclr.cc/paper_files/paper/2026/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html
- **[S07] Huang et al. (2026), “Do LLMs Benefit From Their Own Words?”** arXiv:2602.24287. Assistant-history ablation and context pollution in real conversations. https://arxiv.org/abs/2602.24287
- **[S08] Xiong et al. (2026), “How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior.”** ACL. Retrieved experience induces output similarity; selective add/delete beats naïve growth. https://aclanthology.org/2026.acl-long.27/
- **[S09] Zhang et al. (2026), “Useful Memories Become Faulty When Continuously Updated by LLMs.”** arXiv:2605.12978. Consolidation-induced degradation; recent preprint. https://arxiv.org/abs/2605.12978
- **[S10] “SF-AMS: Strategic Forgetting for Structured Memory in LLM Agent” (2026).** arXiv:2607.22562. Selective forgetting for long-context agents; recent preprint. https://arxiv.org/abs/2607.22562
- **[S11] Li et al. (2025), “DORA: Dynamic Optimization Prompt for Continuous Reflection of LLM-based Agent.”** COLING. Early Stop Reflection and task-adaptive diverse advice. https://aclanthology.org/2025.coling-main.504/
- **[S12] Du et al. (2026), “Escaping the Echo Trap: Enhancing Exploration in Multi-Turn Reinforcement Learning through Turn-Level Credit Assignment.”** ACL. Reflection copying and tree credit; RL-trained setting. https://aclanthology.org/2026.acl-long.1636/

### Research agents, exploration, and creativity

- **[S13] Tang & Yang (2026), “AI Research Agents Narrow Scientific Exploration.”** arXiv:2605.27905. 37,802 ideas, four frameworks, six models; recent preprint. https://arxiv.org/abs/2605.27905
- **[S14] Bhushan, Zhang & Wang (2026), “Can LLM Agents Discover? Evaluating Creativity on ML Engineering Tasks.”** COLM/arXiv:2608.30047. Multi-turn ML-agent P-creativity, H-creativity, feasibility, and impact. https://arxiv.org/abs/2608.30047
- **[S15] Wadinambiarachchi et al. (2024), “The Effects of Generative AI on Design Fixation and Divergent Thinking.”** CHI. Human-AI design fixation and fixation displacement. https://arxiv.org/abs/2403.11164
- **[S16] Chen et al. (2026), “Diversity Collapse in Multi-Agent LLM Systems.”** Findings of ACL. Structural coupling, authority, group size, and topology. https://aclanthology.org/2026.findings-acl.13/
- **[S17] Carbonati et al. (2026), “Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization.”** arXiv:2603.28959. Prompt-coupled strategy collapse and decomposed search. https://arxiv.org/abs/2603.28959

### Human design fixation and recovery

- **[S18] Jansson & Smith (1991), “Design Fixation.”** Design Studies 12(1):3-11. Foundational exemplar-copying experiments. https://www.sciencedirect.com/science/article/pii/0142694X9190003F
- **[S19] Sio, Kotovsky & Cagan (2015), “Fixation or Inspiration? A Meta-Analytic Review of the Role of Examples on Design Processes.”** Design Studies 39:70-99. Forty-three-study synthesis of narrowed variety and quality/novelty gains. https://www.sciencedirect.com/science/article/pii/S0142694X15000290
- **[S20] Crilly & Cardoso (2017), “Where Next for Research on Fixation, Inspiration and Creativity in Design?”** Design Studies 50:1-38. Level-of-analysis and methodological framework. https://www.sciencedirect.com/science/article/pii/S0142694X17300030
- **[S21] Leahy et al. (2020), “Design Fixation From Initial Examples Provided Versus Self-Generated Ideas.”** Journal of Mechanical Design. Self-generated fixation and Design Heuristics intervention. https://doi.org/10.1115/1.4046446
- **[S22] Viswanathan & Linsey (2020), “Design Fixation in Physical Modeling.”** Journal of Mechanical Design. Familiar examples, warnings, and build-test feedback. https://scholarworks.sjsu.edu/faculty_rsca/1570/
- **[S23] Cheng et al. (2014), “Using Partial Photographs to Support Design Ideation.”** Design Studies. Lower-fidelity/partial stimuli reduce fixation while preserving inspiration. https://www.sciencedirect.com/science/article/pii/S0142694X14000283
- **[S24] Viswanathan et al. (2005), “Following the Wrong Footsteps.”** Journal of Experimental Psychology: Learning, Memory, and Cognition. Misleading-example fixation. https://pubmed.ncbi.nlm.nih.gov/16248755/
- **[S25] Koppel & Storm (2014), “Escaping Mental Fixation.”** Memory. Interruption/inhibition and recovery from inappropriate retrieval. https://pubmed.ncbi.nlm.nih.gov/23607286/
- **[S26] Sio et al. (2017), “When Working Hard Does Not Work.”** Thinking & Reasoning. Distributed effort/incubation reduces repeated-concept retrieval. https://pubmed.ncbi.nlm.nih.gov/28028782/

### Diversity, monoculture, and prompt structure

- **[S27] Doshi & Hauser (2024), “Generative AI Enhances Individual Creativity but Reduces the Collective Diversity of Novel Content.”** Science Advances 10. Randomized human story study. https://doi.org/10.1126/sciadv.adn5290
- **[S28] Agarwal, Jin & Matusik (2025), “Investigating the Role of Cognitive Style in LLM-Driven Engineering Design.”** Journal of Computing and Information Science in Engineering. Adaptive-versus-innovative prompt trade-off. https://doi.org/10.1115/1.4066857
- **[S29] Wu, Black & Chandrasekaran (2025), “Generative Monoculture in Large Language Models.”** ICLR. Narrowed output distributions and limits of sampling controls. https://proceedings.iclr.cc/paper_files/paper/2025/hash/5178b2f2d7c44aa390c0777dc77b3f0c-Abstract-Conference.html
- **[S30] Murthy, Ullman & Hu (2025), “Large Language Models Show Human-Like Content Biases in Transmission Chains but Lack Human-Like Conceptual Diversity.”** NAACL. Human/LLM diversity comparison and alignment effect. https://aclanthology.org/2025.naacl-long.561/
- **[S31] Wenger & Kenett (2026), “Large Language Models are Homogeneously Creative.”** PNAS Nexus 5(3). Strong individual originality with low population variability. https://academic.oup.com/pnasnexus/article/5/3/pgag042/8529001
- **[S32] Ueda et al. (2025), “Diversity and Quality in Multi-Agent LLM Ideation.”** SIGDIAL. Cohort size, dialogue depth, persona heterogeneity, and critic diversity. https://aclanthology.org/2025.sigdial-1.26/
- **[S33] Yun et al. (2025), “The Price of Format: Semantic Diversity Collapse from Prompt Templates.”** Findings of EMNLP. Structured format as a diversity suppressor. https://aclanthology.org/2025.findings-emnlp.836/
- **[S34] Si, Yang & Hashimoto (2024/2025), “Can LLMs Generate Novel Research Ideas?”** ICLR. Expert-reviewed novelty, lower feasibility, and low generation diversity. https://arxiv.org/abs/2409.04109

### Agent frameworks, correction, and quality-diversity

- **[S35] Huang et al. (2024), “MLAgentBench.”** ICML/PMLR. Iterative ML experimentation and planning failures. https://proceedings.mlr.press/v235/huang24y.html
- **[S36] Toledo et al. (2025), “AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench.”** arXiv:2507.02554. Search-policy/operator-set interactions. https://arxiv.org/abs/2507.02554
- **[S37] Novikov et al. (2025), “AlphaEvolve.”** arXiv:2506.13131. Population-based LLM code evolution with evaluators. https://arxiv.org/abs/2506.13131
- **[S38] Lu et al. (2025), “Benchmarking Language Model Creativity: A Case Study on Code Generation.”** NAACL. Denial prompting and NEOGAUGE. https://aclanthology.org/2025.naacl-long.141/
- **[S39] Shinn et al. (2023), “Reflexion.”** NeurIPS. Verbal feedback and episodic memory for agent improvement. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
- **[S40] Kamoi et al. (2024), “When Can LLMs Actually Correct Their Own Mistakes?”** TACL. Critical survey of self-correction evidence. https://aclanthology.org/2024.tacl-1.78/
- **[S41] Huang et al. (2024), “Large Language Models Cannot Self-Correct Reasoning Yet.”** ICLR. Limits of intrinsic self-correction. https://openreview.net/forum?id=IkmD3fKBPQ
- **[S42] Chen et al. (2016), “A Survey of Premature Convergence in Evolutionary Computation.”** Information Sciences. Search-policy analogue. https://www.sciencedirect.com/science/article/pii/S002002551500729X
- **[S43] Lehman & Stanley (2011), “Abandoning Objectives: Evolution Through the Search for Novelty Alone.”** Evolutionary Computation 19(2):189-223. Novelty Search. https://pubmed.ncbi.nlm.nih.gov/20868264/
- **[S44] Mouret & Clune (2015), “Illuminating Search Spaces by Mapping Elites.”** arXiv:1504.04909. MAP-Elites and behavioral niches. https://arxiv.org/abs/1504.04909
- **[S45] Shumailov et al. (2024), “AI Models Collapse When Trained on Recursively Generated Data.”** Nature. Training-distribution collapse; distant analogy only. https://www.nature.com/articles/s41586-024-07566-y
- **[S46] Ege et al. (2025), “ChatGPT as an Inventor.”** AI EDAM 39:e6. Forty-eight-hour prototyping study reporting design fixation, abandonment, and complexity. https://doi.org/10.1017/S0890060425000010
- **[S47] Mu et al. (2024), “DDPrompt.”** ACL Short Papers. Prompt ensembles for diverse reasoning paths. https://aclanthology.org/2024.acl-short.17/
- **[S48] Tian et al. (2024), “MacGyver.”** NAACL. Creative physical problem solving; divergent-convergent prompting and reflection. https://aclanthology.org/2024.naacl-long.297/
- **[S49] Tan Min Sen et al. (2026), “Beyond Divergent Creativity.”** Findings of EACL. Human-grounded novelty plus contextual appropriateness. https://aclanthology.org/2026.findings-eacl.138/
- **[S50] “Ask Again, Then Fail” (2024).** ACL. Repeated challenge can make models abandon correct answers; counterweight to persistence-only accounts. https://aclanthology.org/2024.acl-long.577/

### Project evidence

- **[P2] Repository Paper 2 analysis and frozen research notes.** Exact 32-pair state-matched Tiny Addition forks and descriptive Fashion-MNIST replication. `papers/aiscik2026/paper2/analysis.py`; `papers/aiscik2026/RESEARCH_NOTES.md`.

## Search and evidence method

The search followed citation chains and keyword families rather than the word *fixation* alone. Queries covered LLM fixation/anchoring, self-preference, multi-turn history, agent memory and forgetting, reflection collapse, autonomous-research diversity, multi-agent convergence, generative monoculture, design fixation, incubation, novelty search, MAP-Elites, and self-correction. Primary papers, official proceedings, publisher pages, and arXiv records were preferred. Peer-reviewed direct evidence was weighted above preprints and conceptual analogies. Claims in the report are mapped to source IDs in the companion `claim-source-ledger.md`.

The review is broad, not a formal systematic review: it does not claim exhaustive database coverage, dual screening, or publication-bias correction. The term family is too fragmented and the 2026 literature too recent for a stable closed corpus. The evidence map is intended to guide paper framing and experiment design.
