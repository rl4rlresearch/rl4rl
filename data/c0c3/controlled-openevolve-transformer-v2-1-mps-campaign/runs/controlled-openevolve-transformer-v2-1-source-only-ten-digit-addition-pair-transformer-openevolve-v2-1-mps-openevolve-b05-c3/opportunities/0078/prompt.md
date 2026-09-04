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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1313, "training_steps": 4999}
prior_hypothesis: Extending the verified six-coordinate `ln1` scale quotient to seven coordinates will reduce the model from 1,314 to 1,313 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while one adaptive scale preserves optimization flexibility absent from the failed fully non-affine design.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9997, "parameters": 1314, "training_steps": 4999}
prior_hypothesis: Extending the verified five-coordinate `ln1` scale quotient to six coordinates will reduce the model from 1,315 to 1,314 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while two adaptive scales remain.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9989, "parameters": 1305, "training_steps": 4999}
prior_hypothesis: Gauge-fixing one input coefficient in every value-projection row will reduce Reference Design 1 from 1,313 to 1,305 parameters while retaining at least 99% accuracy, because LayerNorm outputs lie in a seven-dimensional zero-sum subspace and the omitted coefficients are exact function-space redundancies.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 1315, "training_steps": 4999}
prior_hypothesis: Extending the verified four-coordinate `ln1` scale quotient to five coordinates will reduce the model from 1,316 to 1,315 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while three adaptive scales remain.

## Recent verification evidence

RECENT RESULT
hypothesis: Starting from the verified 1,328-parameter design, removing the eight-parameter first LayerNorm bias will produce 1,320 learned parameters and retain at least 99% accuracy, because content-independent attention maps that bias to a position-independent output offset absorbable by the attention projection bias, modulo downstream LayerNorm-invariant common shifts.
change: Adopt the verified globally gauge-fixed tied embedding and complete attention-output gauge, then remove `ln1.bias` while retaining learned causal relative-lag attention and gauge-aware optimization.
mechanism: Pre-attention LayerNorm-bias absorption
evidence_used: Reference Design 2 achieved 99.84% accuracy with 1,328 parameters using the global tied-embedding gauge and all eight attention-output gauges; its row-normalized, content-independent attention makes the first LayerNorm bias an exact redundant offset coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965999999999999, "parameters": 1320, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the remaining eight `ln1` scale parameters from the qualified 1,320-parameter design will produce 1,312 learned parameters and retain at least 99% accuracy, because content-independent attention allows the scale to be absorbed exactly into the unrestricted learned value projection.
change: Adopt Reference Design 3’s global tied-embedding, complete attention/MLP projection gauges, and removed `ln1` bias, then make `ln1` entirely non-affine while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware clipping.
mechanism: Pre-attention LayerNorm affine absorption
evidence_used: Reference Design 3 achieved 99.66% accuracy with 1,320 parameters after removing `ln1.bias`; its attention routes are independent of content, so the remaining LayerNorm scale only right-scales the learned value matrix and adds no independent function-space capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4251, "parameters": 1312, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the verified 1,320-parameter design, fixing one `ln1` scale coordinate at one will produce 1,319 parameters and retain at least 99% accuracy, because the learned value projection can absorb that channel scale while the other seven adaptive scales preserve the optimization flexibility lost by the failed fully non-affine design.
change: Adopt the verified global embedding, complete attention/MLP projection, and `ln1`-bias gauges, then quotient only the final `ln1` scale coordinate while preserving full-shape initialization and gauge-aware optimization elsewhere.
mechanism: Single-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The affine-without-bias design achieved 99.66% at 1,320 parameters, whereas removing all eight remaining `ln1` scales collapsed to 42.51%; a one-coordinate quotient directly tests whether seven adaptive scales suffice without repeating the destructive all-at-once reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9952, "parameters": 1319, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified one-coordinate `ln1` scale quotient to two coordinates will reduce the model from 1,319 to 1,318 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb both fixed channel scales while six adaptive scales retain substantially more optimization flexibility than the failed fully non-affine design.
change: Fix the final two `ln1` scale coordinates at one instead of only the final coordinate, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Two-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The one-coordinate quotient achieved 99.52% accuracy at 1,319 parameters, whereas removing all eight scales collapsed to 42.51%; the next single-coordinate extension most directly tests where the optimization-sensitive boundary begins.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9959, "parameters": 1318, "training_steps": 4999}

RECENT RESULT
hypothesis: The unrestricted per-lag lookup is not load-bearing: three learned Gaussian delay components per head will retain at least 99% accuracy while reducing routing parameters from `n_head * (max_seq_len - 1)` to `8 * n_head`, because fixed-format routing should require only a few stationary delay modes.
change: Replace each head’s dense relative-lag table with a learned three-component continuous delay mixture, retaining independent centers and widths per head and gauge-fixed mixture logits; leave token representation, value projection, MLP, checkpoints, and decoding unchanged.
mechanism: Learned Gaussian delay-mixture attention
evidence_used: Content-independent dual-head learned-lag routing reached 99.85%, showing stationary learned routes suffice, while reducing token-representation rank collapsed to 5.06%; this motivates preserving representation capacity while challenging the shared assumption that stationary routing needs an unrestricted logit for every lag.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4985, "parameters": 1292, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-coordinate `ln1` scale quotient to three coordinates will reduce the model from 1,318 to 1,317 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while five adaptive scales remain.
change: Replace the current affine first LayerNorm with a bias-free LayerNorm whose final three scales are fixed at one, leaving attention, optimization, checkpoints, and decoding unchanged.
mechanism: Three-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The two-coordinate quotient achieved 99.59% accuracy with 1,318 parameters, while the one-coordinate quotient achieved 99.52%; removing all eight scales failed at 42.51%, motivating the next incremental coordinate rather than a large reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9947, "parameters": 1317, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified three-coordinate `ln1` scale quotient to four coordinates will produce 1,316 learned parameters and retain at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while four adaptive scales remain.
change: Adopt the qualified global tied-embedding and complete attention/MLP projection gauges, remove `ln1` bias, and fix its final four scale coordinates at one while preserving full-space initialization and gauge-aware optimization.
mechanism: Four-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The three-coordinate quotient achieved 99.47% accuracy with 1,317 parameters, after the one- and two-coordinate quotients also exceeded 99%; this incremental extension directly probes the optimization-sensitive boundary exposed by the failed all-eight-scale removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1316, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified four-coordinate `ln1` scale quotient to five coordinates will reduce the model from 1,316 to 1,315 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while three adaptive scales remain.
change: Fix the final five `ln1` scale coordinates at one instead of the final four, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Five-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The four-coordinate quotient achieved 99.96% accuracy at 1,316 parameters, after the one-, two-, and three-coordinate variants all exceeded 99%; this next one-coordinate extension most directly probes the boundary indicated by the failed all-eight-scale removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1315, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified five-coordinate `ln1` scale quotient to six coordinates will reduce the model from 1,315 to 1,314 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while two adaptive scales remain.
change: Fix the final six `ln1` scale coordinates at one, leaving the learned attention, initialization, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Six-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The five-coordinate quotient achieved 99.97% accuracy with 1,315 parameters after every one-through-four-coordinate quotient also exceeded 99%; this incremental extension most directly locates the boundary suggested by the failed all-eight-scale removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1314, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified six-coordinate `ln1` scale quotient to seven coordinates will reduce the model from 1,314 to 1,313 learned parameters while retaining at least 99% accuracy, because the learned value projection can absorb the additional fixed channel scale while one adaptive scale preserves optimization flexibility absent from the failed fully non-affine design.
change: Fix the final seven `ln1` scale coordinates at one, leaving one learned scale and preserving the learned attention, initialization, optimizer, checkpoints, and decoding interface.
mechanism: Seven-coordinate pre-attention LayerNorm-scale quotient
evidence_used: The six-coordinate quotient achieved 99.97% accuracy with 1,314 parameters, while fixing all eight scales failed at 42.51%; testing seven fixed coordinates is the direct incremental experiment that locates the boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1313, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing one input coefficient in every value-projection row will reduce Reference Design 1 from 1,313 to 1,305 parameters while retaining at least 99% accuracy, because LayerNorm outputs lie in a seven-dimensional zero-sum subspace and the omitted coefficients are exact function-space redundancies.
change: Adopt the verified seven-coordinate `ln1` scale quotient, then remove eight redundant value-projection coefficients while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware clipping.
mechanism: LayerNorm-nullspace value-projection quotient
evidence_used: Reference Design 1 achieved 99.96% accuracy with 1,313 parameters and one adaptive `ln1` scale; complete attention-output and MLP projection gauges also retained accuracy, supporting another exact, optimizer-aware linear-map quotient without repeating the failed removal of the final adaptive scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1305, "training_steps": 4999}

RECENT RESULT
hypothesis: Absorbing `ln2` bias into a restored full `fc1` bias and gauge-fixing one input coefficient in every `fc1` row will reduce the verified model from 1,305 to 1,286 parameters while retaining at least 99% accuracy.
change: Remove eight `ln2` bias parameters, restore the omitted `fc1` bias, remove twelve LayerNorm-nullspace-redundant `fc1` weights, and train those reduced weights with the existing full-space gauge optimizer and clipping.
mechanism: Pre-MLP affine absorption and LayerNorm-nullspace quotient
evidence_used: The current value-projection nullspace quotient removed eight coefficients and achieved 99.89% accuracy; the earlier pre-attention LayerNorm-bias absorption also retained 99.66%. Here bias absorption is direct because `fc1` already supplies pre-GELU biases.
result: the implementation could not be verified



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
