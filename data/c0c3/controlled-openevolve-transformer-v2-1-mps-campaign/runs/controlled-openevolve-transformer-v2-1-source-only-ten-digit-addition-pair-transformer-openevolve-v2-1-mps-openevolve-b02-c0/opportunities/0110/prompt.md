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
verified_results: {"accuracy": 0.9973000000000001, "parameters": 1577, "training_steps": 4999}
prior_hypothesis: Tying the fifth `ln1` scale to learned scale 1 will reduce the model to 1,577 parameters while retaining at least 99% accuracy, with all three verified tie mappings preserved.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Anchoring one input weight in the first key-projection row will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because `ln1` outputs are zero-mean at initialization and key vectors are invariant to sequence-independent shifts.
change: Store seven coordinates for the first key-projection row, reconstruct its final weight as zero, and initialize from the same full 192-scalar draw while projecting out that row’s common input component.
mechanism: Single-key-row normalized-input gauge fixing
evidence_used: The analogous normalized-input gauge removed one `fc1` weight per row and retained 99.93% accuracy; applying it to only one key row is a more conservative independent reduction than the repeatedly unsuccessful fourth `ln1` scale tie.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5322, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fifth `ln1` scale to learned scale 1 will reduce the model to 1,577 parameters while retaining at least 99% accuracy, with all three verified tie mappings preserved.
change: Store four `AnchoredLayerNorm` scales and reconstruct scale 4 from scale 1 while preserving mappings 5→2, 6→0, and 7→1.
mechanism: Partner-specific fourth pre-attention LayerNorm scale tie
evidence_used: The verified three-tie mapping achieved 99.97% at 1,578 parameters. Fourth ties to scales 3, 2, and 0 produced materially different results, making scale 1 the only remaining learned partner for a controlled partner-specific test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1577, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying scale 3 to learned scale 1 will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because scale 1 was the only successful partner for the preceding tie and already controls corresponding scale 7 in the other attention head.
change: Store three `AnchoredLayerNorm` scales and reconstruct scales 3–7 from learned scales 1, 1, 2, 0, and 1, preserving all four verified mappings.
mechanism: Head-coordinate-aligned fifth pre-attention LayerNorm scale tie
evidence_used: The scale-4 tie to scale 1 achieved 99.73% at 1,577 parameters, while ties to scales 0, 2, and 3 failed; extending the uniquely successful partner to the remaining head-aligned coordinate is the closest controlled reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4285, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fifth attention-output projection column to zero mean will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because the removed component is invisible to downstream LayerNorm and initialization preserves the verified model’s function and RNG consumption.
change: Extend `AttentionGaugeLinear` from four to five zero-mean columns and initialize the added gauge by projecting a full eight-coordinate baseline draw.
mechanism: RNG-aligned fifth attention-output common-mode gauge
evidence_used: The 1,577-parameter design achieves 99.73% while already constraining four attention-output columns; the fifth `ln1` tie fell to 42.85%, motivating an adjacent extension of an established independent gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7006, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing scale 3 at its initial value will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because QKV weights can absorb that scale without coupling its gradients into a successfully shared learned scale.
change: Store three learned `AnchoredLayerNorm` scales, fix scale 3 at one, and preserve the verified mappings 4→1, 5→2, 6→0, and 7→1.
mechanism: Fixed-gauge pre-attention LayerNorm scale absorption
evidence_used: The preserved four-tie mapping achieved 99.73% at 1,577 parameters, while additionally tying scale 3 to scale 1 fell to 42.85%; fixing scale 3 isolates whether harmful shared-gradient coupling, rather than loss of its independent parameter, caused that failure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7469, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring the adjacent positional row with coordinate differences will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because initialization changes that row only by an exactly LayerNorm-invisible common shift while avoiding the failed orthogonal Adam reparameterization.
change: Store seven differences for positional row `num_embeddings - 12`, reconstruct its final coordinate as zero, and preserve full-size initialization draws and all existing gauges.
mechanism: Last-coordinate positional common-shift gauge
evidence_used: The orthogonal ninth-row gauge failed at 16.06%, but the verified 1,577-parameter model already contains a successful last-coordinate positional anchor; the analogous difference-coordinate gauge in `NormalizedInputLinear` also retained 99.93%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.18969999999999998, "parameters": 1576, "training_steps": 4999}



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
