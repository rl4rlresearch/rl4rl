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
verified_results: {"accuracy": 1.0, "parameters": 1464, "training_steps": 31999}
prior_hypothesis: Reducing `d_ff` from 7 to 6 will lower the model from 1,480 to 1,464 parameters while retaining at least 99% accuracy when trained for 32,000 steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing independent position vectors with a fixed sinusoidal basis and eight learned channel scales will reduce parameters by `8 * (max_seq_len - 1)` while retaining at least 99% accuracy, because addition’s position relationships are regular and can be represented compositionally without narrowing the successful attention or MLP.
change: Remove the learned positional embedding table; encode absolute positions with generic sinusoidal features whose channel amplitudes are learned from fresh initialization.
mechanism: Shared sinusoidal absolute-position manifold
evidence_used: The 1,596-parameter `d_model=8, d_ff=12` model reached 99.99%, while narrower MLPs and small affine gauge changes failed. This challenges the shared assumption that every position needs an independent learned vector while preserving the demonstrated load-bearing feature capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 1420, "training_steps": 4999}

RECENT RESULT
hypothesis: Compressing only the MLP-input map onto the seven-dimensional mean-zero LayerNorm subspace will reduce parameters from 1,596 to 1,584 while retaining at least 99% accuracy.
change: Add an orthonormal mean-zero input projection and use it only for `fc1`, preserving its bias, output width, initialization scale, and full function class on normalized inputs.
mechanism: Isolated LayerNorm-nullspace compression
evidence_used: Jointly compressing QKV and `fc1` reached 97.76%, close to the threshold, while the unchanged 1,596-parameter model reached 99.99%; isolating the smaller 12-parameter `fc1` reduction tests the exact redundancy with less optimization disruption.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8024, "parameters": 1584, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate at zero will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because an orthogonal rotation of that head’s query and key coordinates can place any query-bias vector in the resulting three-dimensional subspace without changing attention logits.
change: Learn seven query-bias coordinates and pad the final coordinate with zero during attention; leave initialization, widths, training, and decoding unchanged.
mechanism: Single-head query-basis gauge anchoring
evidence_used: The 1,596-parameter design achieved 99.99% accuracy. Although anchoring one MLP-output bias coordinate failed, this tests a distinct exact symmetry internal to one attention head and preserves the current zero initialization with only a one-scalar reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2903, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Gauge-fixing only one QKV input-weight row will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because non-affine LayerNorm makes one weight direction unobservable and gauge-equivalent initialization limits optimization disruption to a single row.
change: Replace the QKV linear map with an equivalent learned map whose first output row uses seven anchored weights while its other 23 rows remain unchanged.
mechanism: Single-row LayerNorm nullspace gauge fixing
evidence_used: Compressing all 36 nullspace directions reached 97.76%, close to the threshold, while the 1,596-parameter model reached 99.99%; testing one of those exact redundancies is the smallest informative step toward locating the safe compression boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1328, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: The exact 36-parameter LayerNorm-nullspace compression will exceed 99% accuracy when trained for 8,000 steps, since it already reached 97.76% at 5,000 steps without narrowing the represented function class.
change: Reparameterize QKV and `fc1` on the seven-dimensional mean-zero LayerNorm subspace, and increase the default training budget from 5,000 to 8,000 steps.
mechanism: Quotient-space compression with extended convergence
evidence_used: The 1,560-parameter joint nullspace design reached 97.76%, much closer to the threshold than other compressed failures, while the uncompressed model reached 99.99%; this suggests convergence, rather than representational capacity, is the most informative variable to test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987, "parameters": 1560, "training_steps": 7999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 12 to 11 will lower the model from 1,560 to 1,544 parameters and still reach at least 99% accuracy when trained for 12,000 steps.
change: Use an 11-unit feed-forward layer and extend the convergence budget from 8,000 to 12,000 steps.
mechanism: One-neuron MLP narrowing with extended convergence
evidence_used: The 1,560-parameter quotient-space model improved from 97.76% at 5,000 steps to 99.87% at 8,000 steps, showing that additional training can overcome slow convergence; this motivates retesting the previously unsuccessful width-11 boundary with a larger budget.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1544, "training_steps": 11999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 11 to 10 will lower the model from 1,544 to 1,528 parameters while retaining at least 99% accuracy when trained for 16,000 steps.
change: Use a 10-unit feed-forward layer and extend the default training budget from 12,000 to 16,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: The quotient-space model reached 99.87% with `d_ff=12` at 8,000 steps, and `d_ff=11` reached 99.93% at 12,000 steps; this supports testing the next width boundary with another 4,000-step convergence allowance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1528, "training_steps": 15999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 10 to 9 will lower the model from 1,528 to 1,512 parameters while retaining at least 99% accuracy when trained for 20,000 steps.
change: Use a 9-unit feed-forward layer and extend the default training budget from 16,000 to 20,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Successive reductions from `d_ff=12` to 11 and then 10 achieved 99.87%, 99.93%, and 99.98% with progressively longer training, supporting the next one-neuron reduction with another 4,000-step allowance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1512, "training_steps": 19999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 9 to 8 will lower the model from 1,512 to 1,496 parameters while retaining at least 99% accuracy when trained for 24,000 steps.
change: Use an 8-unit feed-forward layer and extend the default training budget from 20,000 to 24,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: The quotient-space model remained successful through successive `d_ff` reductions from 12 to 9 with 4,000 additional steps per neuron removed; most recently, `d_ff=9` achieved 100% accuracy at 20,000 steps, motivating the next boundary test with the same convergence allowance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1496, "training_steps": 23999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 8 to 7 will lower the model from 1,496 to 1,480 parameters while retaining at least 99% accuracy when trained for 28,000 steps.
change: Use a 7-unit feed-forward layer and extend the default training budget from 24,000 to 28,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Successive `d_ff` reductions from 12 through 8 all exceeded 99% accuracy when paired with 4,000 additional training steps per removed neuron; most recently, `d_ff=8` achieved 99.99% at 24,000 steps with 1,496 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1480, "training_steps": 27999}

RECENT RESULT
hypothesis: Replacing independent eight-dimensional position vectors with arbitrary six-dimensional learned position codes and a shared learned projection will reduce parameters while retaining at least 99% accuracy after 40,000 steps.
change: Replace the full positional table (`8L` parameters) with a rank-six factorization (`6L + 48` parameters), orthogonally initialized and variance-scaled, while preserving the successful attention and MLP widths.
mechanism: Learned low-rank absolute-position manifold
evidence_used: The current 1,480-parameter model reached 100% accuracy, whereas fixed sinusoidal positions reached 0%; this suggests positional information is load-bearing but does not establish that every position needs an independent eight-dimensional vector. Free per-position codes preserve learned absolute addressing, and prior quotient-space models show that additional training can overcome slower convergence from constrained parameterizations.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"accuracy": 0.9998, "parameters": 1482, "training_steps": 39999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 7 to 6 will lower the model from 1,480 to 1,464 parameters while retaining at least 99% accuracy when trained for 32,000 steps.
change: Use a 6-unit feed-forward layer and extend the default training budget from 28,000 to 32,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Every successive reduction from `d_ff=12` through `d_ff=7` exceeded 99% accuracy when paired with 4,000 additional steps per removed neuron; most recently, `d_ff=7` achieved 100% accuracy at 28,000 steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1464, "training_steps": 31999}



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
