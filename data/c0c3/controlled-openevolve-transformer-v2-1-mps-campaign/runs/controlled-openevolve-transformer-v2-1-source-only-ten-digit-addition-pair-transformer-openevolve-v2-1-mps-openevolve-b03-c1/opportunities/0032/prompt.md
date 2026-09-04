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
verified_results: {"accuracy": 0.9998, "parameters": 1490, "training_steps": 4999}
prior_hypothesis: Centering each harmonic readout row and reconstructing its eighth coordinate as the negative sum of the other seven will reduce the model from 1,494 to 1,490 parameters while retaining at least 99% accuracy, because per-position shifts along the all-ones feature direction are exactly removed by every downstream LayerNorm.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing a second rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,585 to 1,584 while retaining at least 99% accuracy, because a second orthogonal basis rotation preserves both the initialized positional table and latent-factor geometry.
change: Rotate positional latent dimensions 1 and 2 so `pos_proj[1, 0]` is zero, omit that scalar, and reconstruct both fixed zeros during the forward pass.
mechanism: Second orthogonal gauge fixing of the positional factors
evidence_used: The first initialization-preserving positional-factor gauge reduction achieved 99.96% accuracy with 1,585 parameters, directly motivating another one-scalar removal using the same exact invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third rotationally redundant scalar from the rank-four positional factors will reduce parameters from 1,584 to 1,583 while retaining at least 99% accuracy, because another orthogonal latent-basis rotation preserves the initialized positional table and factor geometry exactly.
change: Rotate positional latent dimensions 2 and 3 so `pos_proj[2, 0]` is zero, omit that scalar, and reconstruct all three fixed zeros during the forward pass.
mechanism: Third orthogonal gauge fixing of the positional factors
evidence_used: The first and second initialization-preserving positional-factor gauge reductions both retained 99.95% or better accuracy while reducing one parameter each, directly motivating one more incremental application of the same invariance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1583, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating latent dimensions 0 and 1 to fix `pos_proj[0, 1]` at zero will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy, because the rotation preserves the initialized positional table and the three existing first-column constraints exactly.
change: Apply a fourth initialization-preserving latent-basis rotation, omit the newly fixed projection scalar, and reconstruct the four fixed zeros during each forward pass.
mechanism: Residual orthogonal gauge fixing of rank-four positional factors
evidence_used: Three successive orthogonal positional-factor gauge reductions retained 99.96%, 99.95%, and 99.98% accuracy; after aligning the first projection column, dimensions 0 and 1 retain an unused rotational degree of freedom that can eliminate one entry in the second column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1582, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating latent dimensions 1 and 2 to fix `pos_proj[1, 1]` at zero will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because this orthogonal rotation preserves the initialized positional table and all four existing zero constraints exactly.
change: Apply an initialization-preserving rotation between positional latent dimensions 1 and 2, omit the newly fixed projection scalar, and reconstruct the five fixed zeros during each forward pass.
mechanism: Fifth orthogonal gauge fixing of rank-four positional factors
evidence_used: Four successive orthogonal positional-factor gauge reductions retained at least 99.94% accuracy, including the latest 1,582-parameter model; dimensions 1 and 2 both remain zero in the first projection column, so their residual rotation can eliminate a second-column scalar without disturbing prior constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1581, "training_steps": 4999}

RECENT RESULT
hypothesis: Rotating latent dimensions 0 and 1 to fix `pos_proj[0, 2]` at zero will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because both rows are already zero in the first two projection columns, so this rotation preserves the initialized positional table and all five existing constraints exactly.
change: Apply an initialization-preserving rotation between positional latent dimensions 0 and 1, omit the newly fixed scalar, and reconstruct the six fixed zeros during every forward pass.
mechanism: Sixth orthogonal gauge fixing of rank-four positional factors
evidence_used: Five successive orthogonal positional-factor gauge reductions retained at least 99.94% accuracy, with the latest 1,581-parameter design reaching 99.98%; the remaining rotation between rows 0 and 1 can eliminate a third-column scalar without disturbing prior constraints.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1580, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `pos_proj[3, 0]` at its initialized value will reduce the model from 1,580 to 1,579 learned parameters while retaining at least 99% accuracy, because scaling positional-code column 3 and inversely scaling projection row 3 leaves their product unchanged.
change: Store the initialized positional pivot as a fixed buffer, omit it from the learned flattened projection, and reconstruct the fourth projection row during every forward pass.
mechanism: Positional-factor scaling gauge fixing
evidence_used: Six successive initialization-preserving positional-factor gauge reductions retained at least 99.94% accuracy, including 99.98% at 1,580 parameters; the rank-four factorization still has an exact per-latent-dimension scaling redundancy after exhausting its six orthogonal rotations.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1579, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `pos_proj[2, 1]` at initialization will reduce learned parameters from 1,579 to 1,578 while retaining at least 99% accuracy, because inverse scaling of positional-code column 2 preserves the represented positional table.
change: Store the row-2 projection pivot as a fixed buffer, omit it from the learned flattened projection, and reconstruct it during every forward pass.
mechanism: Second positional-factor scaling gauge fixing
evidence_used: Fixing the analogous row-3 pivot achieved 99.87% accuracy with 1,579 parameters, demonstrating that an initialization-preserving positional scaling gauge can remove a learned scalar without losing required accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1578, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the adaptive rank-four positional lookup with two fixed Fourier harmonics will remove `4 * max_seq_len - 8` learned parameters while retaining at least 99% accuracy, because learned full query/key projections can turn harmonic coordinates into sum- and difference-of-position alignment features, while the relative-bias table continues to provide exact distance cues.
change: Freeze absolute position representation to generic two-frequency sine/cosine coordinates, restore a fully learned 4-by-8 positional readout, and remove the positional-factor reconstruction and its fixed pivots.
mechanism: Fixed harmonic positional coordinates with learned dense readout
evidence_used: Rank-four absolute codes plus relative-distance attention reached 99.97%, showing four positional coordinates suffice; the shared-key collapse to 36.45% argues for preserving the full learned query/key machinery that can interpret the fixed harmonic coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1494, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering each harmonic readout row and reconstructing its eighth coordinate as the negative sum of the other seven will reduce the model from 1,494 to 1,490 parameters while retaining at least 99% accuracy, because per-position shifts along the all-ones feature direction are exactly removed by every downstream LayerNorm.
change: Replace the learned 4-by-8 positional readout with a centered 4-by-7 parameterization and reconstruct the omitted coordinates during the forward pass.
mechanism: LayerNorm positional shift gauge fixing
evidence_used: Fixed Fourier coordinates with a dense learned readout achieved 99.96% accuracy at 1,494 parameters; this preserves that successful positional representation while removing four functionally unobservable LayerNorm shift directions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1490, "training_steps": 4999}



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
