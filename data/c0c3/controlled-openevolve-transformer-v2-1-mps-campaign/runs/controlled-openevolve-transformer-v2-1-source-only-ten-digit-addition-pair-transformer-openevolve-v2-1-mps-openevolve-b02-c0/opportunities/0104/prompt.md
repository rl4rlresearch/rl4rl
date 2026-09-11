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
verified_results: {"accuracy": 0.9997, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Tying the sixth `ln1` scale to the third learned scale while preserving the two successful existing tie mappings will reduce the model to 1,578 parameters and retain at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining the fifth attention projection weight column to zero mean while preserving the verified 1,580-parameter model’s initial function and downstream RNG state will retain at least 99% accuracy with 1,579 parameters.
change: Extend `AttentionGaugeLinear` from four to five zero-mean columns and initialize it by projecting the fifth column from the exact 60-scalar compact draw used by the current four-column design.
mechanism: RNG-aligned fifth attention-output common-mode gauge fixing
evidence_used: A direct fifth attention-column constraint previously failed, while RNG-aligning the failed third `fc2` constraint raised accuracy from 73.46% to 99.89%; this tests the same initialization-preserving remedy on an independent exact downstream-LayerNorm gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2661, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fifth `ln2` scale at one will reduce the model to 1,579 parameters while retaining at least 99% accuracy, because the following learned `fc1` weights can absorb that coordinate’s scale and the initialization remains functionally and RNG-identical.
change: Store three rather than four learned `BiasAnchoredLayerNorm` scales and reconstruct its final five scales as ones.
mechanism: Pre-MLP LayerNorm scale absorption
evidence_used: The verified 1,580-parameter design achieves 99.89% accuracy with four `ln2` scales already fixed at one, while the analogous normalized-input gauge removed one `fc1` weight per row and previously retained 99.93%; this makes one more pre-MLP scale constraint an independent, initialization-preserving reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.927, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Exactly compensating a second reference-token coordinate anchor in the positional embeddings, while preserving the verified 1,580-parameter initialization and RNG consumption, will retain at least 99% accuracy with 1,579 parameters.
change: Remove the second-last reference-token scalar, initialize from the current 111-scalar token draw, translate every token in that coordinate, and absorb the translation into every positional row modulo its LayerNorm-invisible common mode.
mechanism: Function-preserving joint token–position translation gauge
evidence_used: A direct second token-coordinate anchor reached 96.04%, but RNG alignment alone reached only 71.34%; unlike both attempts, this patch applies the full joint token–position gauge transformation and therefore preserves the successful model’s initial computation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7376999999999999, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the final `fc2` weight column to zero mean while preserving the verified 1,580-parameter model’s initial function and RNG consumption will achieve at least 99% accuracy with 1,579 parameters.
change: Keep the first three constrained `fc2` columns, constrain the nonadjacent final column, and initialize all retained coordinates from the same 94-scalar baseline draw used by the successful design.
mechanism: Nonadjacent terminal-MLP common-mode gauge fixing
evidence_used: Three constrained `fc2` columns achieved 99.89%, while constraining the adjacent fourth column fell to 74.27%; testing the final column distinguishes a hidden-unit-specific optimization failure from a general limit on this exact downstream-LayerNorm gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8787999999999999, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fourth `fc2` column while updating it through full-coordinate AdamW moments will produce a 1,579-parameter model with at least 99% accuracy by preserving the verified 1,580-parameter model’s functional optimization trajectory modulo the LayerNorm-invisible common mode.
change: Store the fourth `fc2` column in seven zero-mean coordinates and train those coordinates with a projected AdamW optimizer that maintains the eight coordinatewise moments used by the successful baseline.
mechanism: Optimizer-faithful projected fourth-column gauge
evidence_used: The RNG-aligned fourth-column constraint fell to 74.27% even though the same exact gauge succeeded for three columns at 99.89%, indicating that AdamW’s coordinate-dependent optimization geometry—not representational capacity or initialization—is the most relevant remaining difference.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3857, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying a second `ln1` scale coordinate will reduce the model from 1,580 to 1,579 parameters while retaining at least 99% accuracy, because the unconstrained QKV weight columns can absorb both tied scales and the all-ones initialization remains functionally and RNG-identical.
change: Store six rather than seven learned `AnchoredLayerNorm` scales and reconstruct its final two scales by tying them to the first two learned coordinates.
mechanism: Incremental pre-attention LayerNorm scale absorption
evidence_used: The verified 1,580-parameter design already dynamically ties one `ln1` scale and achieves 99.89%; extending that successful pre-attention gauge by one coordinate tests an untried redundancy after additional `ln1` bias, `ln2` scale, embedding, attention, and `fc2` constraints failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying a third `ln1` scale coordinate will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy, because the learned QKV weight columns can absorb the tied scale and the all-ones initialization remains functionally and RNG-identical.
change: Store five rather than six learned `AnchoredLayerNorm` scales and reconstruct its final three scales by tying them to the first three learned coordinates.
mechanism: Incremental pre-attention LayerNorm scale tying
evidence_used: The immediately preceding extension from one to two dynamically tied `ln1` scales achieved 99.92% accuracy at 1,579 parameters, while several constraints in other components failed; extending the same successful gauge by one coordinate is the closest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.462, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the sixth `ln1` scale to the third learned scale while preserving the two successful existing tie mappings will reduce the model to 1,578 parameters and retain at least 99% accuracy.
change: Store five `AnchoredLayerNorm` scales; reconstruct scale 5 from scale 2 while continuing to reconstruct scales 6 and 7 from scales 0 and 1.
mechanism: Order-preserving third pre-attention LayerNorm scale tie
evidence_used: The two-tie mapping achieved 99.92%, whereas the failed final-three reconstruction remapped both established ties and achieved 46.2%; this isolates the additional tie without disturbing the successful mappings or all-ones initial function.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fifth `ln1` scale to the fourth learned scale while preserving all three verified tie mappings will reduce the model to 1,577 parameters and retain at least 99% accuracy.
change: Store four `AnchoredLayerNorm` scales; reconstruct scale 4 from scale 3 while continuing to reconstruct scales 5, 6, and 7 from scales 2, 0, and 1.
mechanism: Order-preserving fourth pre-attention LayerNorm scale tie
evidence_used: The order-preserving third tie achieved 99.97% at 1,578 parameters, whereas remapping the established ties achieved only 46.2%; extending the successful mapping without disturbing it is the closest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5263, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the immediately preceding positional-embedding row to zero mean will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because per-position common-mode shifts are removed by every LayerNorm and the initialization preserves the verified model modulo that exact invariance.
change: Extend the existing eight-row orthogonal positional gauge to the adjacent ninth row, using the same full initialization draw and basis projection.
mechanism: Adjacent positional common-mode gauge fixing
evidence_used: The verified 1,578-parameter design achieves 99.97% accuracy while already constraining eight adjacent positional rows with this exact gauge; this repeats that established construction instead of extending the fourth `ln1` tie that fell to 52.63%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1606, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fifth `ln1` scale to the third learned scale will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because that learned scale already supports the verified sixth-scale tie without disturbing the three successful mappings.
change: Store four `AnchoredLayerNorm` scales and reconstruct scales 4 and 5 from scale 2 while preserving the verified mappings for scales 6 and 7.
mechanism: Partner-specific fourth pre-attention LayerNorm scale tie
evidence_used: The order-preserving third tie to scale 2 achieved 99.97% at 1,578 parameters, whereas tying scale 4 to scale 3 achieved only 52.63%; changing only the new tie’s partner directly tests whether that failure was partner-specific.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6884999999999999, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fifth `ln1` scale to the first learned scale will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because coordinates 0 and 4 occupy corresponding dimensions in the two attention heads and all three verified tie mappings remain unchanged.
change: Store four `AnchoredLayerNorm` scales and reconstruct scale 4 from scale 0 while preserving scale mappings 5→2, 6→0, and 7→1.
mechanism: Head-aligned fourth pre-attention LayerNorm scale tie
evidence_used: The verified three-tie mapping achieved 99.97% at 1,578 parameters; unsuccessful fourth ties to scales 3 and 2 reached 52.63% and 68.85%, respectively, motivating a controlled test of the architecturally head-aligned scale-0 partner.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0931, "parameters": 1577, "training_steps": 4999}



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
