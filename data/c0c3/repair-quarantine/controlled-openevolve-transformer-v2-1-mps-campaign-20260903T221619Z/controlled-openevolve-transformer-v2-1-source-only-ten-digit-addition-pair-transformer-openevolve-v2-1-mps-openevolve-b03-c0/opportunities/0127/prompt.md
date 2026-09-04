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
verified_results: {"accuracy": 0.9998, "parameters": 1572, "training_steps": 20999}
prior_hypothesis: Gauge-anchoring positional row `max_seq_len // 2 - 1` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates and an implicit zero anchor, preserve its initialization modulo the shift gauge, and optimize it with `GaugeAdamW`.
mechanism: Extended positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, and the malformed `+8` trial still reached 99.44%; a complete deduplicated extension is the most directly supported one-parameter reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it using seven learned relative coordinates and an implicit zero anchor, preserve its initialization modulo the shift gauge, and optimize it with `GaugeAdamW`.
mechanism: Extended positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still achieved 99.44%; the subsequent unverifiable attempts provide no contrary accuracy evidence, making a complete deduplicated extension the most directly supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialization modulo the shift gauge, and optimize it with `GaugeAdamW`.
mechanism: Extended positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still achieved 99.44%; this complete extension directly tests the adjacent row with correct parameter deduplication.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialization modulo the shift gauge, and optimize it with `GaugeAdamW`.
mechanism: Extend the positional-row LayerNorm shift gauge by one adjacent row
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; the malformed `+8` trial still achieved 99.44%, so a complete correctly deduplicated adjacent extension is the most directly supported one-parameter reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialization modulo the shift gauge, and optimize it with `GaugeAdamW`.
mechanism: Extend the positional-row LayerNorm shift gauge by one adjacent row
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still achieved 99.44%; completing the adjacent-row deduplication is the most directly supported one-parameter reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the eight-parameter `+8` row from the base positional embedding, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialized function modulo the exact shift gauge, and train it with `GaugeAdamW`.
mechanism: Extend the positional-row LayerNorm shift gauge to the adjacent `+8` row
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; the malformed `+8` trial still reached 99.44% while retaining eight extra parameters, so completing that adjacent-row deduplication is the most directly supported one-parameter reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Replace the full eight-parameter `+8` positional row with seven learned relative coordinates and an implicit zero anchor, preserving initialization and training it with `GaugeAdamW`.
mechanism: Exact positional-row LayerNorm shift-gauge extension
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; prior `+8` attempts were unverifiable rather than contrary accuracy evidence, so a complete adjacent-row extension remains the most directly supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialized function modulo the exact shift gauge, and optimize it with `GaugeAdamW`.
mechanism: Extended positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; prior `+8` submissions were unverifiable rather than contrary accuracy evidence, while the malformed trial still reached 99.44%, making a complete adjacent-row extension the most informative supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Replace the full eight-parameter `+8` positional row with seven learned relative coordinates and an implicit zero anchor, preserving its initialized function and training it with `GaugeAdamW`.
mechanism: Extend the exact positional-row LayerNorm shift gauge to the adjacent `+8` row
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters. The prior `+8` attempts were unverifiable rather than contrary accuracy evidence, while the malformed trial still reached 99.44%, so a complete adjacent-row extension remains the most directly supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 - 1` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter midpoint-predecessor positional row, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialization, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Gauging positional rows through midpoint `+7` achieved 99.98% accuracy at 1,573 parameters. Targeting the adjacent ungauged `-1` row tests the same exact symmetry while avoiding the repeatedly unverifiable `+8` construction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1572, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 - 2` will reduce the verified model from 1,572 to 1,571 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter midpoint-minus-two positional row, reconstruct it from seven learned relative coordinates with an implicit zero anchor, preserve its initialized function, and optimize it with `GaugeAdamW`.
mechanism: Extend the exact positional-row LayerNorm shift gauge to the adjacent midpoint-minus-two row
evidence_used: Gauge-anchoring the immediately adjacent `max_seq_len // 2 - 1` row achieved 99.98% accuracy with 1,572 parameters; extending the same exact symmetry one row earlier is the most direct supported one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7, "parameters": 1571, "training_steps": 20999}

RECENT RESULT
hypothesis: Gauge-anchoring a third attention output-projection column will reduce the verified model from 1,572 to 1,571 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Replace one dense projection column with seven learned relative coordinates and an implicit zero anchor, initialize it consistently with the existing gauged columns, and optimize it with `GaugeAdamW`.
mechanism: Third attention-output column shift gauge
evidence_used: The verified 1,572-parameter design already gauge-anchors two attention projection columns and achieves 99.98% accuracy; extending that same exact residual/LayerNorm shift symmetry is better supported than retrying the positional midpoint-minus-two gauge that fell to 70%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4036, "parameters": 1571, "training_steps": 20999}



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
