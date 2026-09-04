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
verified_results: {"accuracy": 0.9904000000000001, "parameters": 1271, "training_steps": 4999}
prior_hypothesis: A 1,271-parameter model will achieve at least 99% accuracy because it preserves the twice-verified fourteen-bin positional design while tying only two already quartet-shared MLP hidden biases.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.993, "parameters": 1272, "training_steps": 4999}
prior_hypothesis: A 1,272-parameter model will retain at least 99% accuracy because the verified 1,273-parameter model achieved 99.59%, while sharing the next-sparsest bias affects only fourteen query-key pairs at full context.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9928, "parameters": 1272, "training_steps": 4999}
prior_hypothesis: The 1,272-parameter model will maintain at least 99% accuracy because the identical fourteen-bin sharing design previously achieved 99.3%.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9998, "parameters": 1275, "training_steps": 4999}
prior_hypothesis: A 1,275-parameter model will retain at least 99% accuracy because the verified 1,276-parameter model achieved 99.96%, while sharing the eleventh-farthest bias affects only eleven query-key pairs at full context and preserves head-specific biases at every more frequent distance.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,271-parameter model will retain at least 99% accuracy because the verified 1,272-parameter model achieved 99.3%, while sharing the next-sparsest bias affects only fifteen query-key pairs at full context.
change: Extend cross-head sharing from fourteen to fifteen maximum-distance attention biases, reducing the model by one learned parameter.
mechanism: Cross-head sharing of the fifteen sparsest relative-distance attention biases
evidence_used: The current 1,272-parameter design met the accuracy requirement while sharing fourteen sparsest distance bins; sharing the next-sparsest bin is the smallest directly supported reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9479000000000001, "parameters": 1271, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,237-parameter model will retain at least 99% accuracy because both attention QKV and MLP input projections only receive mean-zero LayerNorm outputs; removing their 36 algebraically inactive input-direction parameters preserves their learned function classes.
change: Replace dense QKV and MLP input maps with learned maps over an orthonormal seven-dimensional basis of the actual LayerNorm output subspace, while preserving the current 13 shared far-distance biases.
mechanism: Orthonormal LayerNorm-subspace projections
evidence_used: Further distance-bias sharing reduced the 1,271-parameter design to 94.79%, showing that head-specific positional biases are load-bearing; the current 1,273-parameter design reached 99.59%, motivating an orthogonal reduction that preserves those biases and removes exact projection redundancies instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8567, "parameters": 1237, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,271-parameter model will retain at least 99% accuracy because it preserves a head-specific bias at the fifteenth-farthest distance—the distinction lost by the failed 94.79% design—while deriving that contrast from the adjacent learned distance instead of allocating an independent parameter.
change: Share the fifteen sparsest base biases across heads, but add the centered head contrast from the nearest fully learned distance to the first shared bin, reducing the current 1,274-parameter model by three parameters.
mechanism: Boundary-contrast extrapolation for sparse attention biases
evidence_used: The 1,272-parameter design sharing fourteen far-distance bins achieved 99.3%, whereas fully sharing the fifteenth bin at 1,271 parameters fell to 94.79%; this motivates retaining its head distinction under a one-dimensional adjacency constraint.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9251999999999999, "parameters": 1271, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,271-parameter model will retain at least 99% accuracy because it preserves the successful 1,272-parameter model’s fourteen shared far-distance biases while removing one orthogonal parameter by tying a single pair of MLP output-bias coordinates.
change: Adopt the verified fourteen-bin attention-bias sharing design and replace the MLP output projection with the existing one-pair-tied mean-zero projection, reducing parameters from 1,275 to 1,271.
mechanism: MLP output-bias pair tying with proven positional capacity preserved
evidence_used: The fourteen-bin 1,272-parameter design achieved 99.3%, while sharing the fifteenth positional bin caused the 1,271-parameter model to fall to 94.79%; this motivates retaining all load-bearing positional distinctions and testing the next reduction in the MLP instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9593, "parameters": 1271, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,271-parameter model will retain at least 99% accuracy because one QKV output row can be parameterized on the seven-dimensional mean-zero LayerNorm subspace without reducing its learned function class, while leaving the verified 1,272-parameter model’s load-bearing positional biases and all other projections unchanged.
change: Replace one of the 24 dense QKV rows with an orthonormally parameterized seven-coordinate row, removing exactly one learned parameter.
mechanism: Single-row LayerNorm gauge quotient
evidence_used: The 1,272-parameter design achieved 99.3%, whereas further positional sharing and MLP output-bias tying failed; the 1,237-parameter experiment supports testing LayerNorm-subspace redundancy more conservatively by quotienting only one projection row instead of all 36 simultaneously.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.763, "parameters": 1271, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,272-parameter model will maintain at least 99% accuracy because the identical fourteen-bin sharing design previously achieved 99.3%.
change: Extend cross-head sharing from thirteen to fourteen maximum-distance biases, removing one learned parameter.
mechanism: Cross-head sharing of the fourteen sparsest relative-distance attention biases
evidence_used: Reference Design 1 verified this exact 1,272-parameter mechanism at 99.3%; all tested 1,271-parameter alternatives fell below the accuracy threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9928, "parameters": 1272, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,271-parameter model will achieve at least 99% accuracy because it preserves the twice-verified fourteen-bin positional design while tying only two already quartet-shared MLP hidden biases.
change: Extend far-distance sharing from twelve to fourteen bins and tie the last two MLP hidden-bias quartets, reducing the verified 1,272-parameter design by one parameter.
mechanism: Adjacent hidden-quartet bias sharing
evidence_used: The fourteen-bin design achieved 99.30% and 99.28%; prior 1,271-parameter failures altered load-bearing positional biases, the MLP output, or QKV parameterization, whereas this tests an untried reduction in the already aggressively shared hidden biases.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9904000000000001, "parameters": 1271, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,270-parameter model will achieve at least 99% accuracy because it preserves the successful fourteen-bin positional design while replacing the 1,271-parameter model’s two hidden-bias values with one shared value.
change: Extend far-distance sharing from eleven to fourteen bins and share one learned bias across all three MLP hidden quartets, reducing the current model by five parameters and the best verified model by one.
mechanism: Global MLP hidden-bias sharing
evidence_used: The 1,271-parameter reference achieved 99.04% after tying its final two hidden-bias quartets; extending that same reduction by one parameter tests the remaining quartet distinction without altering the twice-verified attention design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9661, "parameters": 1270, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,270-parameter model will achieve at least 99% accuracy because it preserves the first-versus-final-quartets bias contrast available in the successful 1,271-parameter design while removing only the shared bias component retained by the failed global-sharing design.
change: Replace the two learned MLP hidden-bias values with one learned centered contrast, assigning one value to the first quartet and its opposite half-value to each tied final quartet.
mechanism: Centered hidden-quartet contrast
evidence_used: The 1,271-parameter design with biases of the form `[a, b, b]` achieved 99.04%, whereas the 1,270-parameter `[a, a, a]` design achieved only 96.61%; this directly motivates retaining the lost quartet contrast while constraining away the orthogonal common component.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9894, "parameters": 1270, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,270-parameter model will achieve at least 99% accuracy because it retains the successful `[a, b, b]` hidden-bias structure and augments the narrowly failing centered contrast with a learned-magnitude-dependent common component.
change: Replace three independent MLP quartet biases with one scalar producing a centered contrast plus a quadratic common shift, reducing the current model by two parameters.
mechanism: Quadratically coupled hidden-quartet bias
evidence_used: The two-parameter `[a, b, b]` design achieved 99.04%, while the one-parameter pure centered contrast reached 98.94%; the small shortfall motivates restoring common-mode variation without adding another learned parameter.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9142, "parameters": 1270, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,269-parameter model will achieve at least 99% accuracy because it preserves the current verified model’s full MLP function class and effective zero-bias initialization while storing its three quartet biases in three otherwise inactive row-constant weight directions.
change: Remove the three explicit MLP hidden-bias parameters and derive each quartet bias from the normalized row sum of one dense weight row; center those carrier rows at initialization so their initial biases remain zero.
mechanism: Orthonormal LayerNorm-gauge bias packing
evidence_used: The current 1,272-parameter model achieved 99.28%. The 1,237-parameter LayerNorm-subspace experiment showed that wholesale projection reparameterization disrupted training despite targeting algebraically inactive directions, motivating reuse of only three such directions while retaining the dense weight layout and its optimization geometry.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8914, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,268-parameter model will retain at least 99% accuracy because the current 1,275-parameter model achieved 99.98%, while its mean-zero residual stream and tied learned embedding can represent the output geometry without seven independently learned final normalization scales.
change: Replace the seven-parameter anchored final LayerNorm with parameter-free LayerNorm, preserving the current robust attention and MLP mechanisms while forcing output discrimination to be represented in the learned residual stream and tied embedding.
mechanism: Parameter-free final feature normalization
evidence_used: Positional sharing and hidden-bias reductions became fragile near 1,271 parameters, whereas the current design has substantially greater accuracy margin. All available designs assume learned anisotropic final rescaling is necessary; this cleanly tests the alternative that normalization should standardize features while the learned transformer and embedding carry the task representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5809000000000001, "parameters": 1268, "training_steps": 4999}



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
