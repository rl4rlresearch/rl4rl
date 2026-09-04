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
verified_results: {"accuracy": 1.0, "parameters": 1305, "training_steps": 49999}
prior_hypothesis: Fixing scale coordinate 4 while restoring learned scale coordinate 5 will reduce the model from 1,306 to 1,305 parameters and retain at least 99% accuracy, because the failed trailing-coordinate reduction uniquely fixed coordinate 5, whose residual channel receives the attention-mean scalar reused by the MLP.

## Recent verification evidence

RECENT RESULT
hypothesis: Learning five final LayerNorm bias coordinates and fixing the remaining three at zero will reduce the model from 1,313 to 1,312 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Reduce the final LayerNorm bias parameter by one coordinate and increase fixed-zero padding from two coordinates to three.
mechanism: Third fixed-coordinate final LayerNorm bias
evidence_used: The two immediately preceding output-only reductions each fixed one additional final LayerNorm bias coordinate and retained 100% accuracy, reaching 1,313 parameters; extending that successful sequence by exactly one coordinate is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1312, "training_steps": 49999}

RECENT RESULT
hypothesis: Learning four final LayerNorm bias coordinates and fixing the remaining four at zero will reduce the model from 1,312 to 1,311 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Reduce the final LayerNorm bias parameter by one coordinate and increase its fixed-zero padding from three coordinates to four.
mechanism: Fourth fixed-coordinate final LayerNorm bias
evidence_used: The three consecutive final LayerNorm bias reductions each removed one parameter while retaining 100% accuracy, most recently reaching 1,312 parameters; extending the same successful output-only restriction by one coordinate is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1311, "training_steps": 49999}

RECENT RESULT
hypothesis: Learning three final LayerNorm bias coordinates and fixing the remaining five at zero will reduce the model from 1,311 to 1,310 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Reduce the final LayerNorm bias parameter by one coordinate and increase its fixed-zero padding from four coordinates to five.
mechanism: Fifth fixed-coordinate final LayerNorm bias
evidence_used: Four consecutive one-coordinate reductions of the final LayerNorm bias retained 100% accuracy, most recently reaching 1,311 parameters; extending this proven output-only restriction by one coordinate is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1310, "training_steps": 49999}

RECENT RESULT
hypothesis: Replacing independent eight-dimensional position vectors with six-dimensional learned codes and a learned projection will retain at least 99% accuracy while removing `2 * INPUT_LEN - 48` parameters.
change: Factor the positional embedding through a six-dimensional learned bottleneck, with an orthogonally initialized trainable projection preserving its initial signal scale.
mechanism: Learned rank-six positional address space
evidence_used: A prior rank-six positional representation achieved 99.98% accuracy. Unlike the failed isolated fixed-coordinate positional quotient, this coordinated factorization preserves a trainable shared orientation and well-scaled representations while directly challenging the assumption that attention routing requires eight independent positional coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7472, "parameters": 1312, "training_steps": 49999}

RECENT RESULT
hypothesis: Learning two final LayerNorm bias coordinates and fixing the remaining six at zero will reduce the model from 1,310 to 1,309 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Reduce the final LayerNorm bias parameter by one coordinate and increase its fixed-zero padding from five coordinates to six.
mechanism: Sixth fixed-coordinate final LayerNorm bias
evidence_used: Five consecutive one-coordinate reductions of the final LayerNorm bias retained at least 99.99% accuracy, most recently reaching 1,310 parameters; extending the same successful output-only restriction by one coordinate is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1309, "training_steps": 49999}

RECENT RESULT
hypothesis: Learning one final LayerNorm bias coordinate and fixing the remaining seven at zero will reduce the model from 1,309 to 1,308 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Reduce the final LayerNorm bias parameter from two coordinates to one and increase its fixed-zero padding from six coordinates to seven.
mechanism: Seventh fixed-coordinate final LayerNorm bias
evidence_used: Six consecutive one-coordinate reductions of the final LayerNorm bias retained at least 99.99% accuracy, with the latest reaching 100% accuracy at 1,309 parameters; extending this successful output-only restriction by one coordinate is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1308, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing the last learned final LayerNorm bias coordinate at zero will reduce the model from 1,308 to 1,307 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Remove the final LayerNorm bias parameter entirely and use an eight-coordinate fixed-zero bias while preserving its learned scale.
mechanism: Fully fixed final LayerNorm bias
evidence_used: Seven consecutive one-coordinate final LayerNorm bias reductions retained at least 99.99% accuracy, with the latest achieving 100% at 1,308 parameters; fixing the remaining coordinate is the smallest direct extension of that successful sequence.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.29109999999999997, "parameters": 1307, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing `attn.proj.bias[1]` as the single final LayerNorm bias will reduce the model from 1,308 to 1,307 parameters while retaining at least 99% accuracy because it preserves a trainable, zero-initialized output shift without coupling it to the sensitive MLP.
change: Remove the dedicated one-coordinate final LayerNorm bias and supply an otherwise unused attention projection-bias coordinate during the final normalization.
mechanism: Cross-layer reuse of the essential final normalization bias
evidence_used: The dedicated final bias achieved 100% accuracy at 1,308 parameters, while fixing it to zero fell to 29.11%, showing that adaptivity is important; the failed third MLP bias sharing coupled the scalar directly into the sensitive MLP, which this output-only reuse avoids.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5495, "parameters": 1307, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing one final LayerNorm scale coordinate at its initialized value of one while retaining the essential learned bias will reduce the model from 1,308 to 1,307 parameters and preserve at least 99% accuracy.
change: Learn seven final LayerNorm scale coordinates, append a fixed-one eighth coordinate, and leave the successful one-coordinate learned bias unchanged.
mechanism: Single fixed-coordinate final LayerNorm scale
evidence_used: The current 1,308-parameter model achieved 100% accuracy after seven incremental output-only bias restrictions, while removing the last adaptive bias collapsed accuracy to 29.11%; this tests an orthogonal, initialization-preserving output-only restriction without sacrificing that essential bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1307, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing a second final LayerNorm scale coordinate at its initialized value of one will reduce the model from 1,307 to 1,306 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Learn six final LayerNorm scale coordinates instead of seven, append two fixed-one coordinates, and preserve the essential one-coordinate learned bias.
mechanism: Second fixed-coordinate final LayerNorm scale
evidence_used: Fixing the first final LayerNorm scale coordinate retained 99.99% accuracy at 1,307 parameters, while removing the remaining learned final bias collapsed accuracy; extending the successful scale restriction by exactly one coordinate is the smallest informative next change.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1306, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing a third final LayerNorm scale coordinate at its initialized value of one will reduce the model from 1,306 to 1,305 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Learn five final LayerNorm scale coordinates instead of six, append three fixed-one coordinates, and preserve the essential one-coordinate learned bias.
mechanism: Third fixed-coordinate final LayerNorm scale
evidence_used: Two consecutive one-coordinate final LayerNorm scale reductions retained at least 99.99% accuracy, most recently achieving 100% at 1,306 parameters; extending the same successful output-only restriction by one coordinate is the smallest informative next change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7589, "parameters": 1305, "training_steps": 49999}

RECENT RESULT
hypothesis: Fixing scale coordinate 4 while restoring learned scale coordinate 5 will reduce the model from 1,306 to 1,305 parameters and retain at least 99% accuracy, because the failed trailing-coordinate reduction uniquely fixed coordinate 5, whose residual channel receives the attention-mean scalar reused by the MLP.
change: Learn final LayerNorm scale coordinates 0–3 and 5, fix coordinate 4 and coordinates 6–7 at one, and preserve the essential learned bias.
mechanism: Noncontiguous final LayerNorm scale restriction
evidence_used: Fixing trailing scale coordinates 6 and 7 retained 100% accuracy at 1,306 parameters, whereas additionally fixing coordinate 5 fell to 75.89%; testing a different third coordinate isolates whether that failure was coordinate-specific.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1305, "training_steps": 49999}



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
