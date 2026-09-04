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
verified_results: {"accuracy": 0.9998, "parameters": 1573, "training_steps": 20999}
prior_hypothesis: Gauge-anchoring initialized positional row `max_seq_len // 2 + 7` will reduce the model from 1,574 to 1,573 parameters while retaining at least 99% accuracy after 21,000 steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the `+8` row from the base positional embedding, reconstruct it from seven learned relative coordinates and an implicit zero anchor, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row shift-gauge extension
evidence_used: The verified contiguous gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; malformed `+8` trials still reached 99.44%, supporting a correctly deduplicated `+8` extension as the most informative next test.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the original eight-parameter `+8` positional row, replace it with seven learned relative coordinates and an implicit zero anchor, reconstruct it during forward passes, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still reached 99.44%; this supports testing the correctly deduplicated extension.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the original eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates plus an implicit zero anchor, and optimize those coordinates with `GaugeAdamW`.
mechanism: Extended positional-row LayerNorm shift gauge
evidence_used: The contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; the malformed `+8` trial still reached 99.44%, indicating that a correctly deduplicated extension is the strongest supported one-parameter reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the original eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned coordinates plus an implicit zero anchor, initialize it from the same original row, and train it with `GaugeAdamW`.
mechanism: Extended positional-row LayerNorm shift gauge
evidence_used: Gauging contiguous midpoint rows through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still reached 99.44% despite retaining eight extra parameters; correctly removing that row is the most directly supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the original eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates plus an implicit zero anchor, and optimize those coordinates with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Gauging contiguous midpoint rows through `+7` achieved 99.98% accuracy at 1,573 parameters, while a malformed `+8` trial still achieved 99.44%; this supports the row’s gauge tolerance and makes correct deduplication the most informative next reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Correctly gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the original eight-parameter `+8` positional row, reconstruct it from seven learned relative coordinates plus an implicit zero anchor, and optimize those coordinates with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; the malformed `+8` trial still reached 99.44%, making a correctly deduplicated extension the strongest supported one-parameter reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the original eight-parameter `+8` positional row, reconstruct it from seven learned relative coordinates with an implicit zero anchor, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters; the malformed `+8` trial still reached 99.44%, supporting a correctly deduplicated adjacent extension.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the original eight-parameter `+8` positional row, reconstruct it from seven learned relative coordinates and an implicit zero anchor, and train those coordinates with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still achieved 99.44%; this supports extending the same proven gauge to the adjacent row with correct deduplication.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates and an implicit zero anchor, initialize it from the original row, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` implementation still achieved 99.44% but retained eight extra parameters; this supports a correctly deduplicated adjacent extension.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates and an implicit zero anchor, initialize those coordinates from the original row, and optimize them with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while a malformed `+8` implementation still reached 99.44% despite retaining eight extra parameters; this makes correct deduplication of the adjacent row the strongest supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates and an implicit zero anchor, initialize it from the original row, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Contiguous midpoint gauges through `+7` achieved 99.98% accuracy at 1,573 parameters, while the malformed `+8` trial still reached 99.44%; this makes a correctly deduplicated adjacent extension the most directly supported reduction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Gauge-anchoring positional row `max_seq_len // 2 + 8` will reduce the verified model from 1,573 to 1,572 learned parameters while retaining at least 99% accuracy after 21,000 training steps.
change: Remove the full eight-parameter `+8` positional row from the base embedding, reconstruct it from seven learned relative coordinates and an implicit zero anchor, initialize it from the original row, and optimize it with `GaugeAdamW`.
mechanism: Adjacent positional-row LayerNorm shift gauge
evidence_used: Gauging contiguous midpoint rows through `+7` achieved 99.98% accuracy at 1,573 parameters; the unsuccessful `+8` submissions supplied no contrary accuracy result, so a complete patch extending the verified construction by exactly one adjacent row is the most informative next test.
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
