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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1285, "training_steps": 4999}
prior_hypothesis: A 1,285-parameter model will retain at least 99% accuracy because the verified 1,286-parameter model achieved 100%, while sharing only the farthest-distance bias preserves an adaptive value and affects just one query-key pair when the context is full.

REFERENCE DESIGN 1
verified_results: {"accuracy": 1.0, "parameters": 1288, "training_steps": 4999}
prior_hypothesis: Replacing independent seven-dimensional absolute position vectors with two learned causal distance-bias tables will retain at least 99% accuracy while reducing the verified 1,403-parameter design by `5 * INPUT_LEN` parameters, because each attention head can learn its own relative routing profile without compressing the independent query, key, or value maps.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1286, "training_steps": 4999}
prior_hypothesis: Anchoring each head’s distance-zero attention bias will reduce the verified 1,288-parameter relative-bias model to 1,286 parameters while retaining at least 99% accuracy, because softmax is invariant to a head-wise constant shift and the anchored parameterization preserves the full attention function class and zero initialization.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1403, "training_steps": 4999}
prior_hypothesis: Tying two learned attention-output bias coordinates will produce a 1,403-parameter model with at least 99% accuracy while preserving all projection weights and an adaptive bias in every output direction.

## Recent verification evidence

RECENT RESULT
hypothesis: Merging two of the three learned MLP bias quartets into one eight-neuron cluster will produce a 1,403-parameter model with at least 99% accuracy while preserving every hidden neuron and learned weight.
change: Replace three quartet-shared MLP biases with two learned biases: one shared across eight neurons and one shared across four.
mechanism: Learned octet-and-quartet MLP threshold sharing
evidence_used: The verified 1,404-parameter design achieved 99.57% accuracy with three learned bias quartets; merging two existing groups removes one scalar along the consistently successful incremental bias-sharing sequence.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9691, "parameters": 1403, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,368-parameter model will retain at least 99% accuracy because the 36 removed qkv and MLP weight directions are annihilated by their preceding affine-free LayerNorms, while all effective projections and the verified three-quartet MLP thresholds remain learnable.
change: Use the verified three-quartet MLP biases, then parameterize the qkv and first MLP projections only on the seven-dimensional mean-zero subspace produced by LayerNorm.
mechanism: LayerNorm-nullspace weight elimination
evidence_used: The 1,404-parameter three-quartet design achieved 99.57%. Shared values fell to 60.09% and shared query-key projections fell to 93.49%, showing effective attention maps must remain independent; this patch preserves those maps and removes only input-weight components that cannot affect mean-zero LayerNorm outputs.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8482, "parameters": 1368, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,403-parameter model will retain at least 99% accuracy because it preserves the verified three-quartet bias design and removes only one functionally null MLP weight direction, leaving the other eleven neuron projections conventionally parameterized.
change: Share MLP biases across three learned quartets and parameterize one MLP input-weight row in the seven-dimensional mean-zero subspace produced by its preceding LayerNorm.
mechanism: Single-neuron LayerNorm-nullspace elimination
evidence_used: The 1,404-parameter three-quartet design achieved 99.57%. Eliminating all 36 LayerNorm-null directions at once fell to 84.82%, motivating a one-row ablation that retains the same effective function class while minimizing the optimization disturbance.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9678, "parameters": 1403, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying two learned final LayerNorm gains in the verified three-quartet MLP design will produce a 1,403-parameter model with at least 99% accuracy, because the shared gain remains adaptive and the initialized model function is unchanged.
change: Use three learned four-neuron MLP bias groups and replace two independent final LayerNorm gains with one shared learned gain.
mechanism: Adaptive final-normalization gain tie
evidence_used: The 1,404-parameter three-quartet design achieved 99.57%. Prior 1,403 failures constrained MLP thresholds or projection weights; this tests a distinct one-scalar reduction while retaining adaptivity, which the successful MLP bias-sharing sequence indicates is preferable to fixing a parameter.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9869, "parameters": 1403, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying two learned attention-output bias coordinates will produce a 1,403-parameter model with at least 99% accuracy while preserving all projection weights and an adaptive bias in every output direction.
change: Replace the attention output projection with a mean-zero linear layer whose final two internal bias coordinates share one learned scalar; leave the MLP and final normalization unchanged.
mechanism: Adaptive attention-output bias tie
evidence_used: The 1,404-parameter three-quartet design achieved 99.57%; prior 1,403 failures altered MLP thresholds, projection weights, or final-normalization gains, motivating an isolated one-scalar test in the previously untested attention-output bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1403, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,402-parameter model will retain at least 99% accuracy because the verified 1,403-parameter design achieved 99.98% with one adaptive attention-output bias pair, and adding a second disjoint pair preserves every projection weight and an adaptive bias in every internal output direction.
change: Share the MLP hidden biases across three learned quartets and parameterize the attention output projection’s seven internal bias coordinates with three independent values and two learned pair-shared values.
mechanism: Two disjoint adaptive attention-output bias ties
evidence_used: The 1,403-parameter design combining three MLP-bias quartets with one attention-output bias tie achieved 99.98%; extending that successful, previously robust mechanism by one disjoint bias tie is the smallest informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9013, "parameters": 1402, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,402-parameter model will retain at least 99% accuracy because it keeps the verified attention-output bias pair and places the additional adaptive tie in the distinct MLP output projection, avoiding the failed second attention-output tie.
change: Use three learned MLP bias quartets and share one internal mean-zero output-bias pair in both the attention and MLP residual projections.
mechanism: Distributed adaptive residual-output bias sharing
evidence_used: The 1,403-parameter design with one attention-output bias tie achieved 99.98%, while adding a second tie in that same projection fell to 90.13%; distributing the next tie to the previously untested MLP output tests whether the failure was attention-coordinate-specific.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9567, "parameters": 1402, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,402-parameter model will retain at least 99% accuracy because the verified 1,403-parameter design reached 99.98%, while anchoring one zero-bias query coordinate removes only a query-key scaling gauge degree of freedom rather than tying effective attention maps.
change: Apply the verified three-quartet MLP biases and single attention-output bias tie, then replace the 192-parameter QKV matrix with a 191-parameter matrix whose query row 1, column 0 coefficient is fixed at its initialization scale.
mechanism: Gauge-anchored query projection with adaptive bias sharing
evidence_used: The 1,403-parameter attention-bias design achieved 99.98%; wholesale query-key sharing failed at 93.49%, motivating a one-scalar gauge anchor that preserves independent query and key coefficients and their effective bilinear attention map.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0001, "parameters": 1402, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,402-parameter model will retain at least 99% accuracy because anchoring the final value-projection row on the mean-zero LayerNorm input removes a functionally redundant coefficient, while an axis-aligned parameterization and distribution-matched initialization avoid the optimization disruption seen with orthonormal nullspace bases.
change: Replace QKV with a 191-parameter projection whose final value row has seven learned coordinates and a zero eighth coordinate, initialized to reproduce the original row’s function on mean-zero inputs.
mechanism: Axis-aligned LayerNorm-nullspace anchoring in one value row
evidence_used: The current 1,403-parameter model achieved 99.98%. Prior LayerNorm-nullspace reductions preserved the theoretical function class but fell to 84.82% globally and 96.78% for one orthonormally parameterized MLP row, motivating a single, axis-aligned reduction in the less routing-sensitive value projection.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6468, "parameters": 1402, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing independent seven-dimensional absolute position vectors with two learned causal distance-bias tables will retain at least 99% accuracy while reducing the verified 1,403-parameter design by `5 * INPUT_LEN` parameters, because each attention head can learn its own relative routing profile without compressing the independent query, key, or value maps.
change: Adopt the verified three-quartet MLP and single attention-output bias tie, remove absolute position embeddings, and inject learned head-specific relative-distance biases directly into causal attention logits.
mechanism: Head-specific learned relative-distance attention
evidence_used: The 1,403-parameter attention-output-bias design achieved 99.98%, whereas sharing query/key projections fell to 93.49%; this preserves all content projections and instead challenges the load-bearing assumption that every absolute position requires a full learned residual-stream vector.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1288, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring each head’s distance-zero attention bias will reduce the verified 1,288-parameter relative-bias model to 1,286 parameters while retaining at least 99% accuracy, because softmax is invariant to a head-wise constant shift and the anchored parameterization preserves the full attention function class and zero initialization.
change: Adopt the verified relative-distance design, including three MLP bias quartets and one attention-output bias tie, while learning only distances 1 through `max_seq_len - 1` and fixing each head’s distance-zero bias to zero.
mechanism: Gauge-anchored head-specific relative-distance attention
evidence_used: The 1,288-parameter head-specific relative-distance design achieved 100% accuracy; this removes only the exact per-head additive gauge redundancy in those successful routing tables, unlike prior reductions that constrained effective projection maps or adaptive coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1286, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,285-parameter model will retain at least 99% accuracy because the verified 1,286-parameter model achieved 100%, while sharing only the farthest-distance bias preserves an adaptive value and affects just one query-key pair when the context is full.
change: Adopt the verified gauge-anchored relative-attention design, then tie the two heads’ maximum-distance bias while retaining independent biases at every shorter causal distance.
mechanism: Cross-head sharing of the farthest relative-distance bias
evidence_used: The head-specific relative-distance model achieved 100% at 1,288 parameters, and exact distance-zero gauge anchoring retained 100% at 1,286; this tests a minimal adaptive tie in the least frequently used routing bin.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1285, "training_steps": 4999}



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
