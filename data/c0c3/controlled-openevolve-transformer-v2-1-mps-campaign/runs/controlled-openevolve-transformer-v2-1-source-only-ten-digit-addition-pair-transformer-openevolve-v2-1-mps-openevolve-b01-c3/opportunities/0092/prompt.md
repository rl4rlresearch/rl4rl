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
verified_results: {"accuracy": 0.9998, "parameters": 1275, "training_steps": 4999}
prior_hypothesis: A 1,275-parameter model will retain at least 99% accuracy because the verified 1,276-parameter model achieved 99.96%, while sharing the eleventh-farthest bias affects only eleven query-key pairs at full context and preserves head-specific biases at every more frequent distance.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1211, "training_steps": 4999}
prior_hypothesis: A 1,211-parameter transformer will achieve at least 99% accuracy because two-dimensional query/key features should suffice to distinguish the small token vocabulary, while the 99.98%-accurate eleven-bin design preserves head-specific positional routing.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1179, "training_steps": 4999}
prior_hypothesis: A 1,179-parameter transformer will achieve at least 99% accuracy because the verified 1,211-parameter design reached 99.93% with two-dimensional queries and keys, leaving margin to test whether one learned scalar per head is sufficient while preserving full-width values and the robust eleven-bin positional routing.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9931, "parameters": 1270, "training_steps": 4999}
prior_hypothesis: A 1,270-parameter model will achieve at least 99% accuracy because adding a small linear common-mode component to the narrowly failing 98.94% centered contrast preserves its useful quartet distinction while avoiding the quadratic coupling’s zero initial common-mode gradient.

## Recent verification evidence

RECENT RESULT
hypothesis: A 1,268-parameter model will retain at least 99% accuracy because the current 1,275-parameter model achieved 99.98%, while its mean-zero residual stream and tied learned embedding can represent the output geometry without seven independently learned final normalization scales.
change: Replace the seven-parameter anchored final LayerNorm with parameter-free LayerNorm, preserving the current robust attention and MLP mechanisms while forcing output discrimination to be represented in the learned residual stream and tied embedding.
mechanism: Parameter-free final feature normalization
evidence_used: Positional sharing and hidden-bias reductions became fragile near 1,271 parameters, whereas the current design has substantially greater accuracy margin. All available designs assume learned anisotropic final rescaling is necessary; this cleanly tests the alternative that normalization should standardize features while the learned transformer and embedding carry the task representation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5809000000000001, "parameters": 1268, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,270-parameter model will achieve at least 99% accuracy because adding a small linear common-mode component to the narrowly failing 98.94% centered contrast preserves its useful quartet distinction while avoiding the quadratic coupling’s zero initial common-mode gradient.
change: Replace the two learned MLP hidden-bias values with one scalar that assigns the first quartet its full value and each tied final quartet negative one-third of that value.
mechanism: Linearly tilted hidden-quartet contrast
evidence_used: The one-parameter centered `[a, -a/2, -a/2]` design reached 98.94%, much closer to the threshold than global sharing at 96.61%; the quadratic common shift fell to 91.42%, motivating a small linear tilt that is trainable from initialization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9931, "parameters": 1270, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will achieve at least 99% accuracy because it preserves the verified 1,270-parameter model’s tilted hidden-quartet bias, while sharing its scalar with an existing zero-initialized MLP output-bias coordinate that receives direct gradients.
change: Replace the three independent hidden-quartet biases with the verified tilted contrast and derive its scalar from the first MLP output-bias coordinate, eliminating three deduplicated parameters without changing the dense projections or positional biases.
mechanism: Direct zero-bias parameter sharing
evidence_used: The independent tilted contrast achieved 99.31% at 1,270 parameters. The prior 1,269-parameter gauge-packing attempt reached only 89.14% after altering carrier-row initialization and optimization geometry; sharing an explicit zero-initialized bias avoids those disruptions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9323, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted hidden-bias design while extending the already successful attention-output bias pair into a triplet.
change: Use one scalar for the final three attention projection bias coordinates and replace the three MLP quartet biases with the verified tilted contrast, removing three parameters total.
mechanism: Extended attention-output bias tying with tilted MLP contrast
evidence_used: The tilted-contrast design achieved 99.31% at 1,270 parameters, and all qualified designs already tolerate one tied attention-output bias pair; unlike the failed 1,269-parameter cross-module sharing experiment, this reduction keeps the MLP and attention parameters independent.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9007999999999999, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted-bias design while storing one final LayerNorm scale in the mean of a QKV row, a direction functionally inactive on mean-zero pre-attention inputs.
change: Adopt fourteen-bin far-distance sharing and the verified tilted MLP bias, then remove one explicit final-normalization scale and derive it from an otherwise inactive QKV row mean without changing dense-weight initialization.
mechanism: LayerNorm scale packed into an attention-row null direction
evidence_used: The tilted fourteen-bin model achieved 99.31% at 1,270 parameters. Earlier gauge packing altered and centered three carrier rows and achieved 89.14%; this tests a single carrier while preserving the original dense QKV layout and initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.937, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will retain at least 99% accuracy because tying the final two already head-shared relative-bias bins removes one parameter while constraining only the single farthest query-key pair per head beyond the penultimate bin.
change: Replace the fourteen independent far-distance scalars with thirteen scalars, reusing the final scalar for both extreme-distance bins.
mechanism: Adjacent extreme-distance attention-bias tying
evidence_used: The current fourteen-bin tilted-bias model achieved 99.31%; unlike failed 1,269-parameter changes that altered load-bearing projections, normalization, or cross-module optimization, this preserves the model everywhere except its sparsest relative-distance distinction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9721, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted hidden-bias design while applying the already successful pair-tied mean-zero bias parameterization independently to the MLP output.
change: Replace the three MLP hidden-quartet biases with the verified one-scalar tilted contrast, then tie one pair of MLP output-bias coordinates, reducing the current model by three parameters.
mechanism: Independent MLP output-bias pair tying
evidence_used: The tilted hidden-quartet contrast achieved 99.31% at 1,270 parameters, and every qualified design already uses pair tying successfully in the attention output projection; unlike failed 1,269-parameter experiments, this neither couples modules nor changes positional biases, normalization, or dense weights.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9331999999999999, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will achieve at least 99% accuracy because it retains the verified 1,270-parameter tilted hidden-bias design while tying only two adjacent final LayerNorm scales, preserving six learned scale degrees of freedom plus the fixed anchor.
change: Adopt the verified one-scalar tilted MLP bias and replace seven independently learned final-normalization scales with six learned scales, reusing the final learned scale for one adjacent coordinate.
mechanism: Single-pair final-normalization scale tying
evidence_used: The tilted-bias model achieved 99.31% at 1,270 parameters. Removing all seven final-normalization scales fell to 58.09%, motivating a conservative one-parameter scale constraint rather than eliminating learned anisotropy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9345, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,269-parameter model will achieve at least 99% accuracy because it preserves the verified 1,270-parameter architecture while removing only one functionally inactive row-constant direction from the final value-projection row.
change: Adopt fourteen-bin far-distance sharing and the verified tilted MLP bias, then parameterize one QKV row in an orthonormal mean-zero basis, reducing its eight weights to seven.
mechanism: Single-row LayerNorm-nullspace quotient
evidence_used: The tilted fourteen-bin model achieved 99.31% at 1,270 parameters. The 1,237-parameter wholesale LayerNorm-subspace projection disrupted training, motivating a conservative one-row quotient whose effective initialization distribution remains isotropic.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.965, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,269-parameter model will achieve at least 99% accuracy because fixing only the farthest relative-attention bias at its zero initialization preserves all more frequently used learned bins and avoids coupling gradients into the penultimate bin.
change: Replace the learned farthest-distance scalar with a fixed zero while retaining thirteen independently learned far-distance biases.
mechanism: Fixed extreme-distance attention bias
evidence_used: The 1,270-parameter model achieved 99.31%, while tying the two extreme-distance bins reached 97.21%; fixing the single farthest bin isolates the reduction to one query-key pair per head instead of constraining and perturbing the more frequently used penultimate bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9439, "parameters": 1269, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,211-parameter transformer will achieve at least 99% accuracy because two-dimensional query/key features should suffice to distinguish the small token vocabulary, while the 99.98%-accurate eleven-bin design preserves head-specific positional routing.
change: Replace four-dimensional query/key vectors with independent two-dimensional vectors while retaining four-dimensional values, and restore the robust eleven-bin far-distance sharing layout.
mechanism: Half-width content-addressing with high-resolution positional attention
evidence_used: The 1,275-parameter eleven-bin design achieved 99.98%, whereas recent 1,269-parameter failures constrained positional biases, normalization, MLP biases, or residual projections. This tests the previously shared assumption that content addressing needs the full value width while preserving those load-bearing mechanisms.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1211, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,179-parameter transformer will achieve at least 99% accuracy because the verified 1,211-parameter design reached 99.93% with two-dimensional queries and keys, leaving margin to test whether one learned scalar per head is sufficient while preserving full-width values and the robust eleven-bin positional routing.
change: Replace full-width QKV with independent one-dimensional query/key projections per head and a full-width value projection, while restoring the verified eleven-bin far-distance sharing layout.
mechanism: Scalar-per-head content addressing
evidence_used: The 1,211-parameter half-width content-addressing design achieved 99.93%, whereas reductions affecting positional biases, normalization, and residual projections repeatedly failed; this isolates the next reduction to content-addressing width while retaining those successful components.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1179, "training_steps": 4999}



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
