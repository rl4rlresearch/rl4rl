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
verified_results: {"accuracy": 0.9997, "parameters": 1453, "training_steps": 4999}
prior_hypothesis: Replacing the eight-dimensional value bias with a seven-dimensional zero-mean attention output bias will reduce learned parameters from 1,454 to 1,453 while retaining at least 99% accuracy, because zero-dropout attention weights sum to one and the centered output projection maps the value bias to only seven observable residual-stream dimensions.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Constraining every MLP input-weight row to the zero-mean subspace will reduce the model from 1,490 to 1,478 parameters while retaining at least 99% accuracy, because `ln2` outputs have exactly zero coordinate sum and therefore the removed component cannot affect the learned function.
change: Store each 12-by-8 `fc1` weight as 12-by-7 coefficients in a fixed orthonormal basis of the zero-mean subspace, preserving the initialized function while removing 12 unobservable parameters.
mechanism: Orthonormal LayerNorm-nullspace parameterization
evidence_used: Centering the harmonic readout along a LayerNorm-null direction reduced parameters to 1,490 while achieving 99.98% accuracy; the same exact invariance applies independently to every `fc1` row because its input comes directly from a non-affine LayerNorm.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1478, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the attention projection and MLP output to the zero-mean feature subspace will reduce parameters from 1,478 to 1,457 while retaining at least 99% accuracy, because their removed all-ones output components only add per-token scalar shifts that all subsequent LayerNorms erase under the fixed zero-dropout configuration.
change: Store the attention output projection as 7-by-8 coefficients and the MLP output weight and bias as 7-dimensional coefficients in fixed orthonormal zero-mean bases, reconstructing their centered 8-dimensional forms during forward passes.
mechanism: Residual-stream LayerNorm shift gauge fixing
evidence_used: Centering the harmonic positional readout along the same LayerNorm-null all-ones direction achieved 99.98% accuracy, and orthonormally removing the LayerNorm-null components of `fc1` retained 99.95% accuracy at 1,478 parameters.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8422, "parameters": 1457, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining only the attention output projection to the zero-mean feature subspace will reduce the model from 1,478 to 1,470 parameters while retaining at least 99% accuracy, because its removed per-token all-ones component is erased by subsequent LayerNorms under the fixed zero-dropout configuration.
change: Store the 8-by-8 attention output projection as 7-by-8 coefficients in a fixed orthonormal zero-mean basis, reconstruct its centered 8-dimensional output during forward passes, and preserve the observable initialized function.
mechanism: Isolated attention-output LayerNorm gauge fixing
evidence_used: The combined attention/MLP output reduction failed at 84.22%, while the 1,478-parameter baseline achieved 99.95%; isolating the smaller eight-parameter attention component directly tests which part of that combined change caused the optimization failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1470, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining only the MLP output bias to the zero-mean feature subspace will reduce the model from 1,470 to 1,469 parameters while retaining at least 99% accuracy, because its removed all-ones component adds only a per-token scalar shift that the final LayerNorm erases under zero dropout.
change: Store the eight-dimensional `fc2` bias as seven orthonormal zero-mean coefficients, reconstruct it during the forward pass, and preserve its initialized observable function.
mechanism: Isolated MLP-output bias gauge fixing
evidence_used: Jointly centering the MLP output weight and bias caused the 1,457-parameter model to fail at 84.22%, while isolated attention-output centering reached 99.93% at 1,470 parameters; isolating the single MLP bias gauge tests whether the prior failure came from constraining the output weights rather than the exact bias redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7956, "parameters": 1469, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining every key-projection row to the zero-mean input subspace will reduce the model from 1,470 to 1,462 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key shift that cancels exactly in the attention softmax.
change: Store the eight key-projection rows as coefficients in the existing seven-dimensional orthonormal zero-mean basis, reconstruct them during forward passes, and preserve the initialized attention function.
mechanism: Softmax-invariant key-weight centering
evidence_used: The analogous LayerNorm-nullspace parameterization of all 12 MLP input rows retained 99.95% accuracy, while isolated attention-output centering retained 99.93%; unlike the failed MLP-bias constraint, this reduction acts directly through an exact softmax-invariant key shift.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1462, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining every value-projection row to the zero-mean input subspace will reduce the model from 1,462 to 1,454 learned parameters while retaining at least 99% accuracy, because the removed component is position-independent after `ln1` and is exactly representable by the existing value bias.
change: Store each 8-by-8 value projection as 8-by-7 coefficients in the existing orthonormal zero-mean basis and reconstruct its centered weight during every forward pass.
mechanism: LayerNorm-nullspace value-weight centering
evidence_used: The analogous key-weight centering reached 100% accuracy at 1,462 parameters, and MLP input-weight centering reached 99.95%; value centering uses the same LayerNorm-null direction while retaining a full value bias to represent its constant contribution.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1454, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the eight-dimensional value bias with a seven-dimensional zero-mean attention output bias will reduce learned parameters from 1,454 to 1,453 while retaining at least 99% accuracy, because zero-dropout attention weights sum to one and the centered output projection maps the value bias to only seven observable residual-stream dimensions.
change: Retain only the six query-bias parameters, learn seven coefficients for a centered attention output bias, and add that bias after the value aggregation and output projection.
mechanism: Direct zero-mean attention output bias
evidence_used: Attention-output centering achieved 99.93% accuracy at 1,470 parameters, establishing that this projection emits only into the seven-dimensional zero-mean subspace; the current successful 1,454-parameter design retains an eight-dimensional value bias whose position-independent effect passes through that centered projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1453, "training_steps": 4999}



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
