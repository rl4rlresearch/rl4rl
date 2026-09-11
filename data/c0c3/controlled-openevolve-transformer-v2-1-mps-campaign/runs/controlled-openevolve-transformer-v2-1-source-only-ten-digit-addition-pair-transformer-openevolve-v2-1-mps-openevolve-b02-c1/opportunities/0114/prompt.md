# Optimize a transformer for 10-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two 10-digit numbers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% accuracy under the fixed verification process. A
smaller implementation is useful only when it meets that accuracy requirement.
Every submitted implementation is trained from a fresh initialization.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must:

- have nonzero trainable parameters;
- contain and use at least one learned causal self-attention module;
- map token inputs to token logits through the learned model;
- train from a fresh initialization during verification;
- write both `checkpoints/best.pt` and a positive-step `checkpoints/last.pt`;
- keep source code unchanged while training; and
- use the protected generic decoding interface exactly as supplied.

Do not implement or embed decimal arithmetic, carry propagation, place-value
rules, digit lookup tables, finite-state addition transitions, fixed answer
rules, or input-dependent Python logic that directly computes the sum. Do not
hide such a solver in model generation, token processing, training, or saved
weights. Do not add dummy or zero-length parameters to disguise a fixed
algorithm as a learned model.

Do not modify protected files. Do not perform post-training state-dictionary
surgery, substitute a different saved model, truncate weights after training,
or report a parameter count that differs from the submitted model.

## Work boundaries

Minimize parameters. Required result: accuracy >= 0.99.
Editable source files: src/model.py, src/train.py.
Results reported after each verification: accuracy, parameters, training_steps.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or verification yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9989, "parameters": 981, "training_steps": 4999}
prior_hypothesis: Tying head 0’s five farthest relative-distance biases will reduce the model from 982 to 981 parameters while retaining at least 99% accuracy, because successive extensions from two through four tied farthest distances achieved at least 99.97%.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing the final three nonreference LayerNorm scale coordinates will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because the verified 985-parameter design already shares the final two scales and this leaves the load-bearing distinct bias coordinate unchanged.
change: Store five rather than six final-LayerNorm scale parameters and reuse the fifth across the final three nonreference residual coordinates.
mechanism: Extended terminal final-LayerNorm scale sharing
evidence_used: The current design achieves 99.98% while tying the final two nonreference scales; extending that isolated constraint by one coordinate is the smallest orthogonal test after complete bias sharing failed, indicating that the first bias contrast should remain independent.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8195, "parameters": 984, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing final-LayerNorm scales for residual coordinates two and three will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient and tolerate zero companion-key contributions, while the sensitive later coordinates remain independent.
change: Store five final-LayerNorm scales instead of six and reuse the third scale for coordinates two and three, preserving the existing scale sharing for coordinates five and six.
mechanism: Mid-pair final-LayerNorm scale sharing
evidence_used: The 989-parameter design achieved 99.98% with the first two companion-key tail coefficients—corresponding to residual coordinates two and three—fixed at zero, and disjoint MLP input pairing achieved 99.93%; unlike the failed terminal scale extension, this change leaves coordinates four through six unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 984, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing final-LayerNorm scales for residual coordinates zero and one will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient, while all scale and bias distinctions implicated by prior failures remain unchanged.
change: Store four final-LayerNorm scales instead of five and reuse the first scale for coordinates zero and one, preserving the existing sharing for coordinates two/three and five/six.
mechanism: Early-pair final-LayerNorm scale sharing
evidence_used: Mid-pair final-LayerNorm scale sharing achieved 99.75% at 984 parameters when aligned with an existing MLP input pair; applying the same isolated constraint to the other early MLP-shared pair is the smallest informative continuation, unlike the failed extension of the sensitive terminal scale group.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3494, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the shared bias of residual coordinates one through six to the reference value will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because it preserves the distinct first-coordinate bias destroyed by the failed complete-sharing change.
change: Store only the first final-LayerNorm bias contrast and reconstruct all remaining coordinates with zero bias.
mechanism: Reference-fixed shared terminal final-LayerNorm bias
evidence_used: Six-way terminal bias sharing achieved 99.98% while complete nonreference sharing collapsed to 14.27%; this isolates the first-coordinate contrast as load-bearing and makes removal of the remaining shared terminal contrast the most informative complementary test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8633, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 1’s third and fourth companion-key tail coefficients will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because both coefficients require learned nonzero contributions and correspond to residual coordinates already successfully paired by the MLP.
change: Replace the two independently stored companion-key coefficients with one shared coefficient, initialize it from their mean, split its gradient across both virtual coordinates, and reproject the virtual key factor after every optimizer update.
mechanism: Shared load-bearing companion-key pair
evidence_used: Fixing the third coefficient caused complete failure and fixing the fourth reached only 92.14%, showing both are load-bearing; meanwhile, disjoint MLP input pairing achieved 99.93%, specifically supporting shared treatment of their corresponding residual coordinates without deleting their learned contribution.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5524, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s three farthest relative-distance biases will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because the verified design already ties its two farthest biases and head 1 tolerates substantially broader far-distance sharing.
change: Store one fewer head-0 relative-bias parameter and reconstruct its three farthest distances with the shared zero reference.
mechanism: Extended head-0 far-distance bias sharing
evidence_used: The 984-parameter design achieved 99.75% while tying head 0’s two farthest distances and using still stronger distance sharing in head 1; extending only the adjacent head-0 boundary is an isolated one-parameter compression that preserves all content projections and final-LayerNorm distinctions implicated by recent failures.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s four farthest relative-distance biases will reduce the model from 983 to 982 parameters while retaining at least 99% accuracy, because extending its tied region from two to three distances achieved 99.98% and head 1 already tolerates broader far-distance sharing.
change: Store one fewer head-0 relative-bias parameter and reconstruct its four farthest distances with the shared zero reference.
mechanism: Extended head-0 far-distance bias sharing
evidence_used: The immediately preceding one-parameter reduction extended head 0’s far-distance sharing from two to three positions and achieved 99.98% accuracy; continuing the same isolated constraint preserves all content projections and LayerNorm distinctions implicated by prior failures.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 982, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s five farthest relative-distance biases will reduce the model from 982 to 981 parameters while retaining at least 99% accuracy, because successive extensions from two through four tied farthest distances achieved at least 99.97%.
change: Store one fewer head-0 relative-bias parameter and reconstruct its five farthest distances with the shared zero reference.
mechanism: Extended head-0 far-distance bias sharing
evidence_used: Extending head 0’s tied region from three to four farthest distances achieved 99.97% at 982 parameters, following 99.98% with three tied distances; this continues the same isolated one-parameter compression without altering content projections or LayerNorm distinctions implicated by prior failures.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 981, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing independent distance-lookup biases with three trainable Gaussian routing bands per head will substantially reduce parameters while retaining at least 99% accuracy, because successful far-distance sharing indicates that routing profiles are structured, while keeping separate kernels and value maps preserves the head specialization whose removal failed.
change: Replace both relative-bias lookup tables with 18 learned amplitude/center/log-width parameters that generate continuous head-specific bias profiles, and optimize them directly with AdamW rather than the lookup-table quotient update.
mechanism: Head-specific learned radial relative-position routing
evidence_used: Extending head 0’s tied far-distance region from two through five positions retained at least 99.89% accuracy, while sharing value maps across heads collapsed accuracy to 15.71%; this supports compressing positional routing into independent low-dimensional kernels without sharing the heads’ learned content representations.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0209, "parameters": 966, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy, because each successive extension from two through five tied distances remained above 99.89%.
change: Store one fewer head-0 relative-bias parameter and reconstruct its six farthest distances with the shared zero reference.
mechanism: Extended head-0 far-distance bias sharing
evidence_used: The immediately preceding extension to five tied distances achieved 99.89% at 981 parameters, after extensions to three and four distances achieved 99.98% and 99.97%; continuing this isolated constraint is the smallest evidence-backed compression.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy; evaluating every 500 steps will allow the previously timed-out configuration to finish all 4,999 training steps.
change: Remove one head-0 relative-bias parameter, reconstruct six farthest distances with the zero reference, and reduce intermediate validation frequency without changing optimization steps.
mechanism: Extended head-0 far-distance bias sharing with reduced validation overhead
evidence_used: Successive extensions through five tied distances retained at least 99.89% accuracy, while the six-distance attempt timed out rather than producing negative accuracy evidence; its negligible model-side compute change makes reducing validation overhead the most direct retry.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy; validating only at initialization and the final step will eliminate enough evaluation overhead for the previously timed-out configuration to complete all 4,999 updates.
change: Remove one head-0 relative-bias parameter, reconstruct its six farthest distances with the zero reference, and set the default evaluation interval equal to the training length so checkpoints are evaluated at step 0 and step 4,999.
mechanism: Six-distance head-0 bias sharing with final-only validation
evidence_used: Successive extensions through five tied distances retained at least 99.89% accuracy. Both six-distance attempts timed out without negative accuracy evidence, including a retry at 500-step validation intervals, so a stronger reduction in validation overhead is the most direct informative test.
result: training did not finish within the verification time limit



Use the available technical evidence to choose the most informative next
change. Treat unsuccessful or malformed work as evidence when a useful
subject-level reason is provided. Do not invent missing evidence.

## Response

Return these short metadata lines followed by one or more exact
`SEARCH`/`REPLACE` blocks that together produce one implementation:

`MECHANISM: <a concise free-form name for the computational idea>`

`HYPOTHESIS: <a falsifiable claim grounded in the evidence above>`

`INTENDED_EDIT: <what this patch changes>`

`EVIDENCE: <the most relevant prior result and why it motivates this patch>`

Start each block with `<<<<<<< SEARCH`, put the exact existing lines next, use a
line containing `=======` as the divider, put the replacement lines after it,
and finish the block with `>>>>>>> REPLACE`.

Every `SEARCH` section must be nonempty and match exactly once after earlier
blocks have been applied. All blocks must apply. They may edit either or both
editable files, but together they must describe one implementation ready for
verification. The mechanism name is descriptive, not chosen from a fixed list.
Do not paste whole files, lengthy logs, or routine progress reports outside the
patch.
