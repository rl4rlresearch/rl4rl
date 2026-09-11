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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1613, "training_steps": 4999}
prior_hypothesis: Fixing second-LayerNorm bias coordinate 2 will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its constant downstream contribution can be absorbed by `fc1.bias`.

## Recent verification evidence

RECENT RESULT
hypothesis: Fixing feature 3 alongside feature 4 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because both offsets are transferred into positional embeddings and the generalized virtual AdamW path reconstructs the omitted full-model gradients and updates.
change: Generalize the successful feature-4 embedding quotient to features 3 and 4, including initialization transfer, virtual-gradient recovery, clipping, and optimizer projection.
mechanism: Second joint token–position common-column gauge
evidence_used: The feature-4 quotient achieved 99.93% at 1616 parameters; among remaining coordinates, feature 3 has the strongest unused coordinate-specific evidence because its additional LayerNorm-scale gauge narrowly missed the threshold at 98.67%, whereas tested feature-0 and feature-1 embedding extensions failed substantially.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9545999999999999, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `fc2.weight[4, 0]` and subtracting its value from every output row will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because it changes each residual by only a token-dependent uniform feature shift that the final LayerNorm removes, while a virtual AdamW coordinate preserves full-gradient optimization.
change: Extend the coordinate-4 MLP-output gauge to one weight-column coordinate, reconstruct the canonical full weight during forward passes, and add virtual-gradient, clipping, and optimizer handling for the omitted weight.
mechanism: MLP output common-column gauge erased by final LayerNorm
evidence_used: The current 1616-parameter model achieves 99.93% while retaining the successful coordinate-4 MLP-output bias gauge and virtual AdamW treatment. This tests the corresponding activation-dependent output-shift quotient after additional token-position gauges proved accuracy-sensitive.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4578, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the retained value-bias coordinate to output-projection bias coordinate 5 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because it preserves the critical learned value-bias pathway while the direct output-bias contribution keeps the shared direction well-conditioned.
change: Remove the separate value-bias parameter, use `proj_bias[5]` for both value and output bias, and optimize the shared attention bias with ordinary AdamW.
mechanism: Shared value/output-bias coordinate
evidence_used: Removing the remaining value bias collapsed accuracy to 16.83%, whereas the current design achieves 99.93%; sharing that coordinate tests the redundant bias pathway without eliminating the empirically essential value-side signal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.625, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing key-projection coordinate 3 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because this softmax-invisible direction uses the same virtual AdamW quotient already supporting five removed key coordinates.
change: Add key row 3 to the gauge-fixed QKV rows; the existing reconstruction, gradient clipping, and optimizer logic automatically handles the additional omitted coordinate.
mechanism: Sixth LayerNorm-null key-projection gauge
evidence_used: The five-coordinate key quotient is present in the 99.93% design. Among the remaining key coordinates, coordinate 3 has the strongest evidence: its token-position gauge reached 95.46% and its LayerNorm-scale gauge reached 98.67%, while the tested key coordinate 1 reached only 71.24%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2488, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing feature 3 alongside feature 4 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy because maintaining and updating gauge-equivalent full token and positional tensors reproduces full-parameter AdamW exactly, avoiding the projected optimizer dynamics of the 95.46% feature-3 attempt.
change: Remove the final-token feature-3 coordinate, transfer its offset into positional embeddings, and replace projected token–position updates with latent full-tensor AdamW followed by exact gauge reduction.
mechanism: Exact latent full-embedding AdamW under a second token–position gauge
evidence_used: The existing feature-4 quotient achieved 99.93%, while feature 3 was the strongest attempted second feature at 95.46%; its much smaller deficit than other second-feature attempts makes optimizer fidelity on that exact quotient the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0658, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm scale coordinate 3 while restoring its omitted gradient and AdamW moment through the downstream `fc1` affine gauge will reduce the model to 1615 parameters and improve the prior 98.67% result to at least 99%.
change: Remove LayerNorm scale coordinate 3, then jointly optimize the reduced scale and downstream `fc1` parameters using a virtual fifth scale coordinate, gauge-projected updates, and gauge-aware gradient clipping.
mechanism: Virtual-AdamW LayerNorm-scale gauge projection
evidence_used: Fixing coordinate 3 previously reached 98.67%, the closest reported 1615-parameter attempt; coordinates 0, 1, and 4 are already fixed in the 99.93% design, so restoring optimizer dynamics for coordinate 3 is the most targeted follow-up.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7492, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `qkv.weight[4, 7]` will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because the omitted query-weight direction is exactly absorbed by `q_bias[4]`, with virtual AdamW moments preserving optimization of the full affine map.
change: Add query row 4 to the QKV quotient, reconstruct its omitted gradient using LayerNorm scale, LayerNorm bias, and query-bias gradients, and jointly project QKV and query-bias AdamW updates.
mechanism: LayerNorm-null query-weight/q-bias affine gauge
evidence_used: The current feature-4 embedding, LayerNorm-scale, LayerNorm-bias, and MLP-output gauges support 99.93% accuracy. Unlike the failed removal of an essential value bias or an additional key coordinate, this change retains the complete learned query affine map through an exact bias-coupled LayerNorm-null gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.41009999999999996, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 1 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because this bias contribution is absorbable by the learned `fc1` bias and feature 1 has previously supported 99.98% value-bias and 99.81% LayerNorm-scale reductions.
change: Remove LayerNorm-2 bias coordinate 1 in addition to the existing fixed coordinate 4, reconstructing both as zeros during the forward pass.
mechanism: Second LayerNorm-bias/downstream-affine gauge
evidence_used: The verified 1616-parameter design already succeeds with LayerNorm-2 bias coordinate 4 fixed. Among alternative coordinates, feature 1 has the strongest stated affine-coordinate evidence, while recent feature-1 failure concerned the more optimization-sensitive token-position embedding quotient.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 1 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its contribution can be absorbed by the learned `fc1` bias.
change: Remove LayerNorm-2 bias coordinate 1 alongside coordinate 4 and reconstruct both as zeros during forward passes.
mechanism: Second LayerNorm-bias/downstream-affine gauge
evidence_used: The identical prior proposal was not evaluated because its search text matched multiple locations. Coordinate 1 remains the strongest untested choice: related value-bias and LayerNorm-scale reductions achieved 99.98% and 99.81%, while the verified 1616-parameter model already fixes LayerNorm-2 bias coordinate 4.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8711, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 0 will reduce the model to 1615 parameters while retaining at least 99% accuracy, because its contribution is absorbable by `fc1.bias` and coordinate 0 previously tolerated two related reductions above the accuracy threshold.
change: Remove LayerNorm-2 bias coordinate 0 alongside the existing fixed coordinate 4 and reconstruct both as zeros during forward passes.
mechanism: Second LayerNorm-bias/downstream-affine gauge at coordinate 0
evidence_used: Coordinate 0 achieved 99.83% under the related LayerNorm-scale gauge and 99.94% under the analogous value-bias reduction; this makes it the strongest evidence-backed alternative after coordinate-1 LayerNorm-bias fixing reached only 87.11%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9915, "parameters": 1615, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 3 will reduce the verified model from 1615 to 1614 parameters while retaining at least 99% accuracy, because its contribution remains absorbable by `fc1.bias`.
change: Remove LayerNorm-2 bias coordinate 3 alongside coordinates 0 and 4, reconstructing all three as zeros during forward passes.
mechanism: Third LayerNorm-bias/downstream-affine gauge
evidence_used: Coordinate 3 produced the closest simple 1615-parameter failure at 98.67% through the related LayerNorm-scale gauge, while fixing LayerNorm-2 bias coordinate 0 succeeded at 99.15%; this makes coordinate 3 the strongest evidence-backed next bias quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1614, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing second-LayerNorm bias coordinate 2 will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because its constant downstream contribution can be absorbed by `fc1.bias`.
change: Remove LayerNorm-2 bias coordinate 2 alongside coordinates 0, 3, and 4, reconstructing all four as zeros during forward passes.
mechanism: Fourth LayerNorm-bias/downstream-affine gauge
evidence_used: The current three-coordinate bias quotient achieved 99.86% at 1614 parameters. Coordinate 2 is the most informative untested bias coordinate, while coordinate 1 already has direct negative evidence at 87.11%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1613, "training_steps": 4999}



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
