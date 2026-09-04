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
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1585, "training_steps": 4999}
prior_hypothesis: Removing one rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,586 to 1,585 while retaining at least 99% accuracy, because an orthogonal basis change preserves the initialized positional table and factor-space geometry while fixing one projection entry to zero.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the eight zero-initialized attention output-projection bias parameters will reduce the model from 1,612 to 1,604 parameters while retaining at least 99% accuracy, because the retained value bias can supply a position-independent vector through the output projection.
change: Construct the attention output projection normally to preserve initialization ordering, then remove only its learned bias.
mechanism: Remove value-bias-representable attention output bias
evidence_used: The 1,612-parameter model achieved 99.96% accuracy. Although removing value bias failed, that result motivates preserving the optimization-critical value bias and testing the converse redundant parameter: the downstream projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9937, "parameters": 1604, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the explicit eight-parameter query bias will reduce the model from 1,604 to 1,596 parameters while retaining at least 99% accuracy, because the learned `ln1_bias` can produce query offsets through the query projection, the retained value bias can compensate its value-side effect, and its key-side effect is softmax-invariant.
change: Retain only the value portion of the RNG-aligned QKV bias and stop adding a separate query bias.
mechanism: Absorb query offset into the learned pre-attention bias
evidence_used: The 1,604-parameter design achieved 99.37% accuracy; removing value bias collapsed accuracy to 30.85%, while removing `ln1` bias reached only 75.82%, motivating preservation of both and elimination of the query bias they can jointly represent.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9561, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the MLP output bias at zero will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy, because any omitted bias vector is equivalent up to an all-coordinate shift erased by the final LayerNorm.
change: Preserve the full 12-unit MLP and initialization ordering, but learn only seven `fc2` bias coordinates and pad the eighth with zero during the forward pass.
mechanism: Final-LayerNorm bias gauge fixing
evidence_used: The 1,604-parameter design achieved 99.37%, while removing an entire feed-forward unit caused severe accuracy loss; this removes only a one-dimensional LayerNorm-invariant redundancy without narrowing the capacity-critical MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9823999999999999, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one positional-embedding coordinate at zero through an initialization-aligned token/position reparameterization will reduce the model from 1,604 to 1,603 parameters while retaining at least 99% accuracy.
change: Absorb position zero’s first coordinate into the corresponding coordinate of every token embedding, compact the positional parameter by one scalar, and reconstruct the fixed zero coordinate during the forward pass.
mechanism: Single-coordinate token–position embedding gauge fixing
evidence_used: The current 1,604-parameter model achieved 99.37% accuracy, and RNG-aligned removal of an exact attention redundancy previously succeeded where a non-aligned version failed; this patch likewise preserves the initialized model function exactly while testing a different one-dimensional gauge after the MLP-bias gauge reached only 98.24%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1603, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the successful positional-embedding gauge fixing from one coordinate to two will reduce parameters from 1,603 to 1,602 while retaining at least 99% accuracy.
change: Absorb position zero’s first two coordinates into every token embedding, remove both scalars from the positional parameter, and reconstruct them as zeros during the forward pass.
mechanism: Second-coordinate token–position embedding gauge fixing
evidence_used: The initialization-aligned removal of the first positional scalar achieved 99.92% accuracy with 1,603 parameters, strongly motivating the adjacent one-scalar extension of the same exact gauge.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the successful positional-embedding gauge fixing from two coordinates to three will reduce parameters from 1,602 to 1,601 while retaining at least 99% accuracy.
change: Absorb position zero’s first three coordinates into every token embedding, remove those three positional scalars, and reconstruct them as zeros during the forward pass.
mechanism: Third-coordinate token–position embedding gauge fixing
evidence_used: The initialization-aligned one- and two-coordinate gauge fixes achieved 99.92% and 99.75% accuracy respectively, directly motivating one more incremental removal of the same exact redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7373000000000001, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing positional coordinate 7 alongside the already successful coordinates 0 and 1 will reduce parameters from 1,602 to 1,601 while retaining at least 99% accuracy, indicating that the failed coordinate-2 extension was an optimization-sensitive coordinate choice rather than a general three-gauge capacity limit.
change: Absorb position zero’s coordinates 0, 1, and 7 into the corresponding token-embedding coordinates, omit those three positional scalars, and reconstruct them as zeros during the forward pass.
mechanism: Nonadjacent positional-embedding gauge fixing
evidence_used: Fixing coordinates 0 and 1 achieved 99.75% with 1,602 parameters, whereas adding adjacent coordinate 2 collapsed to 73.73%; testing a nonadjacent coordinate is the most direct way to distinguish coordinate-specific optimization sensitivity from failure of every 1,601-parameter positional gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9281, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate at zero will reduce the model from 1,602 to 1,601 parameters while retaining at least 99% accuracy, because the learned pre-attention bias and retained value bias can represent query offsets, and preserving seven query-bias coordinates avoids the optimization collapse seen when all eight were removed.
change: Store seven learned query-bias coordinates plus all eight value-bias coordinates, reconstructing the final query-bias coordinate as zero during attention.
mechanism: Single-coordinate query-bias gauge fixing
evidence_used: Removing all eight query-bias parameters still reached 95.61%, substantially closer to the requirement than removing the value bias or `ln1` bias; this motivates testing the smallest possible one-coordinate reduction while retaining nearly all of the optimization-sensitive query-bias parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second query-bias coordinate at zero will reduce the model from 1,601 to 1,600 parameters while retaining at least 99% accuracy, because the learned pre-attention bias and full value bias can represent query offsets and six explicit query-bias coordinates should preserve most of the optimization benefit.
change: Store six learned query-bias coordinates plus all eight value-bias coordinates, reconstructing the final two query-bias coordinates as zeros during attention.
mechanism: Two-coordinate query-bias gauge fixing
evidence_used: Fixing one query-bias coordinate achieved 99.94% accuracy with 1,601 parameters, while removing all eight still achieved 95.61%; an incremental second-coordinate removal directly tests how much of the remaining query-bias parameterization is optimization-critical.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing full 8-dimensional positional embeddings with learned rank-4 codes while adding per-head relative-distance attention biases will reduce the model below 1,600 parameters and retain at least 99% accuracy, because addition primarily needs relational alignment between output and operand positions rather than eight independent absolute-position coordinates.
change: Factorize the learned absolute positional table to rank 4 using an RNG-preserving truncated SVD initialization, and augment each causal attention head with a learned bias indexed by query-key distance.
mechanism: Low-rank absolute codes with head-specific relative-distance attention
evidence_used: Positional gauge results were sharply coordinate-sensitive—two removed scalars achieved 99.75%, while a third collapsed to 73.73%—and shared-key attention collapsed to 36.45%. This motivates preserving full query/key/value content projections while moving positional alignment into a direct learned relative-attention mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing each head’s distance-zero relative bias to zero will reduce parameters from 1,588 to 1,586 while retaining at least 99% accuracy, because subtracting that bias from every causal attention logit leaves the softmax unchanged.
change: Store only nonzero-distance relative biases and prepend a fixed zero reference bias during attention.
mechanism: Per-head relative-bias softmax gauge fixing
evidence_used: The rank-4 positional/relative-attention design achieved 99.97% accuracy at 1,588 parameters; unlike riskier capacity reductions, this removes one exact softmax-invariant degree of freedom per head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1586, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,586 to 1,585 while retaining at least 99% accuracy, because an orthogonal basis change preserves the initialized positional table and factor-space geometry while fixing one projection entry to zero.
change: Rotate the first two positional latent dimensions so `pos_proj[0, 0]` is zero, omit that scalar from the learned parameter, and reconstruct the fixed zero during inference and training.
mechanism: Orthogonal gauge fixing of the low-rank positional factorization
evidence_used: The rank-four positional/relative-attention model achieved 99.97%, and exact relative-bias gauge fixing retained 99.96%; this tests another exact redundancy within the successful low-rank positional parameterization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1585, "training_steps": 4999}



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
