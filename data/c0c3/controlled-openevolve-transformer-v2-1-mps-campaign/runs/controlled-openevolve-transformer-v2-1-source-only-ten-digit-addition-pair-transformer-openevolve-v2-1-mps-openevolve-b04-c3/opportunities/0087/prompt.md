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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 1.0, "parameters": 1147, "training_steps": 44999}
prior_hypothesis: Sharing one learned key and value projection across both attention heads, while retaining distinct query projections, will achieve at least 99% accuracy after 45,000 updates with 1,147 parameters.

REFERENCE DESIGN 1
verified_results: {"accuracy": 1.0, "parameters": 1144, "training_steps": 44999}
prior_hypothesis: The verified 1,145-parameter model will retain at least 99% accuracy after 45,000 updates when a second coordinate of the tied six-dimensional token code is fixed at zero, yielding 1,144 learned parameters.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1212, "training_steps": 44999}
prior_hypothesis: Fixing a seventh scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,213 to 1,212 learned parameters.

REFERENCE DESIGN 3
verified_results: {"accuracy": 1.0, "parameters": 1143, "training_steps": 44999}
prior_hypothesis: Fixing a third scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,144-parameter design to 1,143 learned parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing a fourth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,216 to 1,215 learned parameters.
change: Replace the absolute-position embedding with the qualified Fourier synthesizer, learn 60 of its 64 mixer weights, and fix the final four weights at zero while preserving full-matrix initialization RNG consumption.
mechanism: Four-entry positional-mixer constraint
evidence_used: Reference Design 3 achieved 100% accuracy with three fixed positional-mixer scalars and 1,216 parameters; the preceding one- and two-scalar constraints also achieved 100%, motivating the smallest incremental fourth constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1215, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a fifth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,215 to 1,214 learned parameters.
change: Adopt the qualified seven-entry query-weight gauge and Fourier positional backbone, then learn 59 of 64 positional-mixer weights with the final five fixed at zero while preserving full-matrix initialization RNG consumption.
mechanism: Five-entry positional-mixer constraint
evidence_used: The four-entry positional-mixer design achieved 100% accuracy with 1,215 parameters, following 100% results for one through three fixed entries; this motivates the smallest incremental fifth constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1214, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a sixth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,214 to 1,213 learned parameters.
change: Learn 58 of the 64 positional-mixer weights, fixing the final six entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Six-entry positional-mixer constraint
evidence_used: The five-entry positional-mixer design achieved 100% accuracy with 1,214 parameters, following 100% results for one through four fixed entries; this motivates the smallest incremental sixth constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1213, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a seventh scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,213 to 1,212 learned parameters.
change: Learn 57 of the 64 positional-mixer weights, fixing the final seven entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Seven-entry positional-mixer constraint
evidence_used: Reference Design 1 achieved 100% accuracy with six fixed positional-mixer scalars and 1,213 parameters, following successful one-through-five-scalar constraints; this motivates the smallest incremental seventh constraint while retaining dense cross-channel mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1212, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing an eighth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,212 to 1,211 learned parameters.
change: Learn 56 of the 64 positional-mixer weights, fixing the final eight entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.
mechanism: Eight-entry positional-mixer constraint
evidence_used: Reference Design 3 achieved 100% accuracy with seven fixed positional-mixer scalars and 1,212 parameters, following successful one-through-six-scalar constraints; this motivates the smallest incremental eighth constraint while retaining learned cross-channel positional mixing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1211, "training_steps": 44999}

RECENT RESULT
hypothesis: Sharing one learned key and value projection across both attention heads, while retaining distinct query projections, will achieve at least 99% accuracy after 45,000 updates with 1,147 parameters.
change: Challenge the assumption that each routing head needs independent key/value features. Adopt the qualified eight-entry Fourier constraint and replace independent per-head K/V projections with one shared learned K/V head, preserving distinct queries and full-matrix initialization RNG consumption.
mechanism: Two-query multi-query attention with a shared learned key/value dictionary
evidence_used: The 1,211-parameter eight-entry Fourier design achieved 100% accuracy, while restricting positional mixing and reducing token-code dimension failed at 46.49% and 56.64%. This preserves those load-bearing representations and instead compresses attention: both routing heads process the same token-position stream, so distinct queries can select different sources while sharing their learned key/value representation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1147, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a ninth scalar in the dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,147 to 1,146 learned parameters.
change: Learn 55 of the 64 positional-mixer weights, fixing the final nine entries at zero while preserving full-matrix initialization RNG consumption and the qualified multi-query attention backbone.
mechanism: Nine-entry positional-mixer constraint
evidence_used: The current 1,147-parameter multi-query design achieved 100% accuracy, and every incremental positional-mixer constraint from one through eight fixed scalars also achieved 100%; testing one additional scalar is the smallest evidence-backed reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3392, "parameters": 1146, "training_steps": 44999}

RECENT RESULT
hypothesis: The verified 1,147-parameter multi-query model will retain at least 99% accuracy when a second attention-output bias scalar is fixed at zero, yielding 1,146 learned parameters after 45,000 updates.
change: Adopt the qualified shared key/value attention design with distinct queries, retain the successful eight-scalar Fourier constraint, and learn six rather than seven attention-output bias coordinates while preserving constructor RNG consumption.
mechanism: Shared-key/value multi-query attention with a two-coordinate output-bias constraint
evidence_used: Shared key/value attention achieved 100% accuracy with 1,147 parameters. The ninth positional-mixer constraint failed at 33.92%, so the next one-parameter probe targets a different redundancy; every qualified backbone already fixes one attention-output bias coordinate without loss.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1146, "training_steps": 44999}

RECENT RESULT
hypothesis: The verified 1,146-parameter multi-query model will retain at least 99% accuracy when a third attention-output bias coordinate is fixed at zero, yielding 1,145 learned parameters after 45,000 updates.
change: Adopt the qualified shared key/value attention and eight-scalar Fourier constraint, then learn five rather than six attention-output bias coordinates while preserving constructor RNG consumption.
mechanism: Three-coordinate attention-output bias constraint on shared-key/value multi-query attention
evidence_used: The 1,146-parameter design achieved 100% accuracy with two fixed attention-output bias coordinates, while a ninth positional-mixer constraint failed at 33.92%; the smallest evidence-backed next probe is one additional output-bias constraint without further reducing positional mixing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0944, "parameters": 1145, "training_steps": 44999}

RECENT RESULT
hypothesis: The qualified 1,146-parameter multi-query model will retain at least 99% accuracy after 45,000 updates when one coordinate of the tied six-dimensional token code is fixed at zero, yielding 1,145 learned parameters.
change: Adopt the qualified shared-key/value attention, eight-scalar Fourier constraint, and two-coordinate attention-output-bias constraint, then learn all but one scalar of the tied token embedding while preserving full initialization RNG consumption.
mechanism: Tied-token residual-basis gauge fixing
evidence_used: The 1,146-parameter multi-query design achieved 100% accuracy, while further reducing positional mixing or attention-output bias failed; this tests a different redundancy, fixing one coordinate under the remaining orthogonal basis freedom of the tied token subspace.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1145, "training_steps": 44999}

RECENT RESULT
hypothesis: The verified 1,145-parameter model will retain at least 99% accuracy after 45,000 updates when a second coordinate of the tied six-dimensional token code is fixed at zero, yielding 1,144 learned parameters.
change: Learn all but two scalars of the tied token embedding, reconstructing both fixed trailing scalars as zeros while preserving full embedding initialization RNG consumption.
mechanism: Two-scalar tied-token residual-basis gauge fixing
evidence_used: Fixing one tied-token scalar achieved 100% accuracy with 1,145 parameters, whereas further positional-mixer and attention-output-bias constraints failed; the smallest informative next probe is one additional constraint in the successful token-code mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1144, "training_steps": 44999}

RECENT RESULT
hypothesis: Fixing a third scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,144-parameter design to 1,143 learned parameters.
change: Adopt the verified two-scalar token constraint and fix one additional trailing embedding scalar at zero, preserving full embedding initialization RNG consumption and tied input/output use.
mechanism: Three-scalar tied-token residual-basis constraint
evidence_used: Reference Design 1 achieved 100% accuracy with two fixed token-embedding scalars and 1,144 parameters, after the one-scalar version also achieved 100%; an incremental third constraint is the most direct evidence-backed reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1143, "training_steps": 44999}



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
