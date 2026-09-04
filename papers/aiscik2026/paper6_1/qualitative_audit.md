# Paper 6 qualitative audit: assumption-challenge checkpoints

## Scope and method

This audit covers the saved public agent messages at every tenth proposal in
the three greedy OpenEvolve 2.1 campaigns used by Paper 6. At each checkpoint,
the analysis pairs the assumption-challenge trajectory with the ordinary
trajectory in the same task, block, and memory stratum. The machine-generated
corpus is `derived/checkpoint_message_corpus.md`; it contains the mechanism,
hypothesis, intended edit, evidence, and evaluator outcome for all 608 matched
checkpoint opportunities. Final agent messages exist for 605; three provider
failures consumed an opportunity without producing a message.

The audit does not treat a rationale as private reasoning. It asks what the
saved proposal publicly claims, whether the patch operationalizes that claim,
what evidence it cites, and what happened when the evaluator ran it. A
deterministic high-information sample adds the largest source departure,
largest retained improvement, and first invalid challenged proposal in each
task-by-memory stratum. I also read complete messages and patches around three
productive checkpoints and their descendants: addition block 4,
single-incumbent, proposals 68--75; Fashion-MNIST block 3,
single-incumbent, proposals 8--15; and nanoGPT block 3, four-lineage,
proposals 8--15.

## Cross-task findings

### 1. The prompt usually changes the *kind* of public proposal

At challenged checkpoints, messages name a load-bearing assumption and a
contrasting representation or computation. Ordinary messages more often
continue the nearest successful local operator. This is not merely the
required phrase appearing in the response: Fashion-MNIST proposals replace
fixed-position heads with spatial pooling or expose differential image bases;
nanoGPT proposals introduce token-conditioned gates, lexical bypasses, or
different global-context allocations; addition proposals replace absolute
positions with learned relative lags, separate routing width from value width,
or bottleneck the token interface while keeping the transformer width.

The code agrees with the prose in the inspected patches. For example,
Fashion-MNIST block 3 proposal 10 registers fixed Sobel/Laplacian kernels,
concatenates their responses with raw pixels, and expands the first
convolution. Addition block 4 proposal 70 actually restricts tied token
embeddings to six learned channels while leaving the residual computation at
eight. NanoGPT block 3 proposal 10 adds a query-conditioned, per-head gate to
the attention output. These are executable changes in the proposed mechanism,
not restyled summaries of the incumbent.

### 2. The intervention trades feasibility for larger departures

The most dramatic alternatives often fail. Fashion-MNIST repeatedly proposes
content-adaptive pooling, multi-scale fusion, learned mixed downsampling, and
spatial recalibration; many exceed runtime or regress. In addition, broad
multi-query sharing, narrow routing, positional factorization, and recurrent
reuse often lose the 99% qualification threshold. In nanoGPT, the evaluator
usually runs successfully, but many larger alternatives fail to improve the
strict incumbent.

This is a process result, not an implementation accident: the prompt asks the
agent to leave the local basin. A strict greedy selector then rejects most
departures. The intervention therefore increases information-producing tests
while lowering immediate retention, especially in Fashion-MNIST.

### 3. Some successful challenges create exploitable descendant programs

The strongest qualitative pattern is a two-stage sequence: one challenged
proposal changes the representation; ordinary descendants then tune or prune
inside the new representation.

- In addition block 4, proposals 68--69 remove one QKV gauge coordinate at a
  time. Proposal 70 challenges the full-width token-interface assumption and
  removes 227 parameters at once. Ordinary proposals 71--72 reduce the token
  bottleneck from six to five to four dimensions, proposal 73 establishes the
  three-dimensional failure boundary, and proposals 74--75 resume safe gauge
  reductions on the four-dimensional design.
- In Fashion-MNIST block 3, proposals 8--9 refine translation augmentation and
  ensembling. Proposal 10 changes the input representation to a fixed
  differential basis and gains 31 correct classifications. The next ordinary
  proposals test multiscale filters and weight averaging; proposal 13 retains
  a 25% endpoint/EMA blend. The prompt opened a new representational branch,
  while ordinary search extracted and combined follow-up gains.
- In nanoGPT block 3, proposals 8--9 tune a logit softcap. Proposal 10 adds
  query-conditioned attention-head gating and is retained. Proposals 11--15
  test full-state, stratified, head-aligned, MLP-branch, and fused variants.
  Several fail, but the descendants are targeted mechanism experiments rather
  than a return to softcap-only tuning.

These cases show why immediate retention alone can understate a particular
checkpoint's scientific effect. They are illustrative rather than typical:
exact recursive parent tracing finds descendants after only 25 challenged
cycles in addition, 10 in Fashion-MNIST, and 5 in nanoGPT. The complete-record
analysis therefore reports policy-window gains separately from exact
descendant gains and does not credit every later proposal to the checkpoint.

### 4. Population conclusions depend on what is coded

Within trajectories, challenged messages are more novel relative to their own
prior mechanism descriptions. Across trajectories, the apparent direction
depends on the semantic representation. Family tags extracted from the full
rationale give lower challenged dispersion in all three tasks. The same
taxonomy applied only to the dedicated mechanism field gives higher challenged
dispersion in all three tasks; a first-mentioned primary-family coding does so
as well. Full rationales contain the old assumption, rejected alternatives,
evidence, and longer contrastive explanation, so their shared vocabulary is not
a clean measure of candidate-mechanism concentration.

The late Fashion-MNIST trace still shows a repeated spatial theme. After early
gains, challenged checkpoints revisit variants of learned pooling,
anti-aliasing, spatial gating, multiscale fusion, and full-grid heads. The
messages use accumulated negative evidence to preserve aspects that proved
load-bearing, but many alternatives differ inside that theme. This is useful
trace evidence of repetition, not proof that the challenged population
converges: mechanism-only coding finds the challenged population more
dispersed than the ordinary one.

### 5. Evidence use improves in form, not necessarily in causal quality

Challenged messages more often contain explicit causal language and compare an
old assumption with a new alternative. The better messages use failed results
diagnostically: full key/value sharing failed, so a later addition proposal
shares one value channel while preserving private channels; fixed Fourier
positions failed, so a later proposal learns head-specific relative lags;
global pooling failed on Fashion-MNIST, so later proposals retain the flattened
spatial head while adding a complementary branch.

But rationales can also explain almost any new edit after the fact. The
repeated Fashion-MNIST mixed-pooling proposals are a warning: citing persistent
translation sensitivity does not make a repeatedly unsuccessful family newly
informative. Evaluator traces, source changes, and descendants must therefore
remain primary evidence; rhetorical compliance is only a manipulation check.

## Task-specific interpretation

### Trained transformer for 10-digit addition (80 proposals)

Ordinary checkpoints are already technically sophisticated: exact symmetry
quotients, LayerNorm-null directions, and single-coordinate pruning can involve
large patches. That is why challenged checkpoints do not show the clear AST or
changed-line jump seen on the other tasks. The content changes nonetheless:
challenge messages are much more likely to propose positional mechanisms,
decoupled routing/content widths, shared value alphabets, token bottlenecks,
and recurrent or factorized computations. Their immediate retention is lower,
but successful challenges produce much larger parameter reductions. The most
compelling evidence is the block-4 six-dimensional token bottleneck followed
by ordinary reductions to four dimensions, and the head-specific relative-lag
mechanism that removes the absolute-position table.

### Fashion-MNIST classifier (200 proposals)

This task shows the cleanest structural manipulation and the sharpest
feasibility cost. Controls at checkpoints often tune loss, ensembling,
calibration, or augmentation. Challenges almost always rewrite spatial
representation or pooling and are much larger in source and AST distance.
Only 5% of challenged checkpoints are retained versus 36.5% of ordinary
checkpoints. Yet rare challenged successes matter: fixed differential inputs,
parallel multi-scale context, and identity-initialized channel attention seed
productive descendants. Over 200 proposals, the repeated prompt also becomes
formulaic and often returns to mixed max/mean pooling. This task supplies the
strongest evidence for local redirection, feasibility cost, and the difference
between within-run repetition and population-level diversity.

### Fixed-time language-model pretraining (40 proposals)

The 40-proposal horizon yields only four challenged checkpoints per treated
trajectory, so estimates are less precise. Challenges clearly enlarge code
changes and shift mechanisms away from scalar schedule/context-window tuning
toward learned gating, lexical residual paths, context allocation, and
alternative feed-forward computation. They rarely improve immediately, but
ordinary descendants after a successful gate or context proposal test a rich
local neighborhood. The block-3 gating sequence is the clearest example.
Because challenged runs were already worse at proposal 9 and remain slightly
worse at proposal 40 in absolute objective, their larger within-trajectory
improvement should not be called endpoint superiority.

## Claim boundaries for the paper

- Strongest claim: at scheduled checkpoints, the instruction reliably changes
  public proposal content and executable source structure on Fashion-MNIST and
  nanoGPT, with weaker source-structure evidence but clear semantic redirection
  on addition.
- Strong process claim: the redirection costs more output tokens and usually
  lowers immediate retention; a small number of successful alternatives yield
  productive ordinary descendants.
- Measurement claim: local departure is robust, but population dispersion
  reverses between full-rationale and mechanism-focused representations. The
  present evidence therefore supports neither a general convergence nor a
  general divergence claim across trajectories.
- Descriptive claim only: challenged trajectories accrue more objective
  improvement from proposal 9 to the task-specific horizon. The trajectories
  had already diverged before treatment, and nanoGPT challenged trajectories
  do not beat controls in absolute final objective.
- Memory condition: single-incumbent versus four-lineage memory is a moderator,
  not the main experiment. No stable cross-task interaction justifies claiming
  that portfolio memory amplifies or suppresses the prompt generally.
