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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1136, "training_steps": 4999}
prior_hypothesis: The 1,136-parameter transformer will maintain at least 99% accuracy because the verified 1,137-parameter design reached 99.92%, and this ties only one pair of initially identical MLP output-bias coordinates while preserving every learned weight matrix and attention mechanism.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9992, "parameters": 1137, "training_steps": 4999}
prior_hypothesis: The 1,137-parameter transformer will achieve at least 99% accuracy because each head’s factorized 8→4→7 value/output path can be replaced by a more expressive direct 8→7 learned map while preserving attention, positional routing, normalization, and residual biases.

REFERENCE DESIGN 2
verified_results: {"accuracy": 1.0, "parameters": 1145, "training_steps": 4999}
prior_hypothesis: The resulting 1,145-parameter transformer will retain at least 99% accuracy because the verified 1,146-parameter model reached 99.99%, and this removes only its zero-initialized scalar MLP bias while preserving the successful attention, normalization, projections, and positional routing.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9979, "parameters": 1080, "training_steps": 4999}
prior_hypothesis: A 1,080-parameter transformer will maintain at least 99% accuracy because the value map receives non-affine LayerNorm outputs, whose all-ones input component is always zero, so removing that unidentifiable input direction preserves its effective learned function class.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 1,171-parameter transformer will achieve at least 99% accuracy because the verified 1,179-parameter scalar-query/key model achieved 100%, while sharing only the eight-parameter key projection preserves head-specific queries, values, and positional routing.
change: Replace full-width QKV with one scalar query per head, one scalar key projection shared across heads, and the existing full-width value projection.
mechanism: Head-shared scalar key addressing
evidence_used: The 1,179-parameter design reached 100% accuracy with one-dimensional queries and keys; unlike unsuccessful reductions to positional biases, normalization, or residual projections, this isolates an eight-parameter reduction to source-token content addressing while retaining distinct head behavior through queries and relative biases.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1171, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,163-parameter transformer will maintain at least 99% accuracy because the verified 1,171-parameter model achieved 100% with a head-shared scalar key, while distinct relative biases and value channels can still specialize the two heads when the scalar query is also shared.
change: Replace the two-head scalar query projection with one shared scalar query projection, relying on the existing broadcast to both heads and removing eight learned parameters.
mechanism: Head-shared scalar query addressing
evidence_used: The current 1,171-parameter design reached 100% after sharing the key projection across heads; this symmetric next reduction preserves separate learned query and key functions, full-width values, and the proven eleven-bin positional routing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1163, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,155-parameter transformer will achieve at least 99% accuracy because the verified 1,163-parameter model reached 99.99% with head-shared scalar queries and keys, leaving one scalar content feature that can serve both roles while positional routing and full-width values remain unchanged.
change: Replace the separate head-specific scalar query/key projection with one eight-parameter projection shared across both heads and both addressing roles.
mechanism: Tied scalar query-key content addressing
evidence_used: The 1,163-parameter design achieved 99.99% after independently sharing query and key projections across heads; tying those two remaining scalar projections is the next isolated reduction and leaves the proven eleven-bin relative biases, full-width values, MLP, normalization, and residual projections intact.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1155, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,148-parameter transformer will achieve at least 99% accuracy because the verified 1,155-parameter model reached 100% with one tied scalar query-key feature, and the learned embeddings can align that feature with a fixed input coordinate while a learned gain preserves adaptive attention strength.
change: Replace the eight-parameter tied scalar address projection with a one-parameter gain applied to the first normalized feature coordinate, while preserving shared query-key addressing, full-width values, and eleven-bin positional routing.
mechanism: Fixed-direction scalar content addressing
evidence_used: The 1,155-parameter tied scalar query-key design achieved 100% accuracy after successive reductions of addressing width and head-specific projections all succeeded, motivating removal of the remaining learned address-direction degrees without altering the proven positional, value, MLP, normalization, or residual components.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1148, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,146-parameter transformer will achieve at least 99% accuracy because the 1,148-parameter fixed-direction scalar-addressing design reached 100%, while the current one-scalar tilted MLP bias independently reached 99.31%.
change: Replace full-width QKV projections with the verified learned scalar gain on one normalized coordinate plus a full-width value projection, restore eleven-bin positional routing, and retain the current one-scalar tilted MLP bias.
mechanism: Fixed-direction scalar addressing with tilted quartet bias
evidence_used: The 1,148-parameter reference achieved 100% using fixed-direction scalar addressing; retaining the current verified tilted bias removes two additional parameters without modifying its successful attention mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1146, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 1,145-parameter transformer will retain at least 99% accuracy because the verified 1,146-parameter model reached 99.99%, and this removes only its zero-initialized scalar MLP bias while preserving the successful attention, normalization, projections, and positional routing.
change: Remove the learned tilted hidden-quartet bias and make the MLP input projection bias-free, reducing the model by one parameter.
mechanism: Bias-free MLP hidden expansion
evidence_used: The current 1,146-parameter design achieved 99.99%; prior reductions to relative biases, final normalization, and projection subspaces failed, so the remaining isolated MLP hidden-bias scalar is the most informative next target.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1145, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,137-parameter transformer will achieve at least 99% accuracy because each head’s factorized 8→4→7 value/output path can be replaced by a more expressive direct 8→7 learned map while preserving attention, positional routing, normalization, and residual biases.
change: Adopt the verified bias-free MLP and fuse the attention value and output weights into direct per-head mean-zero output maps, removing eight factorization parameters.
mechanism: Unfactorized headwise value-output maps
evidence_used: The verified 1,145-parameter bias-free design achieved 100% accuracy; prior failures targeted positional biases or normalization, whereas this patch preserves those mechanisms and removes only a low-rank factorization whose direct replacement has fewer parameters and no smaller effective linear function class.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1137, "training_steps": 4999}

RECENT RESULT
hypothesis: The 1,136-parameter transformer will maintain at least 99% accuracy because the verified 1,137-parameter design reached 99.92%, and this ties only one pair of initially identical MLP output-bias coordinates while preserving every learned weight matrix and attention mechanism.
change: Adopt the verified fixed-direction scalar addressing, direct per-head 8→7 attention output maps, and bias-free MLP expansion, then reduce one additional parameter by pair-tying the mean-zero MLP output bias.
mechanism: Pair-tied MLP output bias with unfactorized headwise attention maps
evidence_used: Reference Design 1 achieved 99.92% with 1,137 parameters. The successful 1,145-parameter result also showed that removing an initially zero MLP hidden bias preserved 100% accuracy, motivating another isolated MLP-bias reduction instead of altering the repeatedly sensitive positional biases or final normalization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1136, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,087-parameter transformer will maintain at least 99% accuracy because the two positional heads can share one learned token-value representation while a seven-parameter diagonal adapter preserves head specialization.
change: Replace independent headwise 8→7 attention maps with one shared 8→7 map and an anchored diagonal adapter for the second head; also adopt the verified fixed-direction addressing and bias-free, pair-tied MLP.
mechanism: Shared value codebook with head-specific diagonal adaptation
evidence_used: The 1,136-parameter design achieved 99.99% with independent direct headwise maps. This tests its unchallenged assumption that each positional router needs an entirely separate value map while preserving the repeatedly load-bearing relative biases, normalization, MLP width, and two-head routing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1087, "training_steps": 4999}

RECENT RESULT
hypothesis: A 1,080-parameter transformer will maintain at least 99% accuracy because the value map receives non-affine LayerNorm outputs, whose all-ones input component is always zero, so removing that unidentifiable input direction preserves its effective learned function class.
change: Replace the shared 8→7 value projection with a learned 7→7 projection over an orthonormal mean-zero input basis, removing seven parameters while preserving the head adapter and all routing mechanisms.
mechanism: Mean-zero input factorization for the attention value map
evidence_used: The current 1,087-parameter design achieved 99.98% accuracy; unlike prior failed reductions to positional biases or normalization, this removes only value weights multiplying a direction eliminated by the existing pre-attention LayerNorm.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1080, "training_steps": 4999}



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
