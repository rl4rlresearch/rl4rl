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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1004, "training_steps": 4999}
prior_hypothesis: A 1,004-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions to eleven, ten, and nine units achieved 100%, 99.89%, and 99.96%, respectively; reducing the verified nine-unit design by one more unit tests the same isolated 14-parameter reduction.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 853, "training_steps": 4999}
prior_hypothesis: The resulting 853-parameter transformer will maintain at least 99% accuracy because the qualified 854-parameter endpoint-anchored design achieved 99.97%, while tying only the two sparsest, farthest relative-distance biases preserves embedding rank, MLP width, attention gains, and all established scale contrasts.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9997, "parameters": 852, "training_steps": 4999}
prior_hypothesis: The resulting 852-parameter transformer will maintain at least 99% accuracy because the qualified 853-parameter design achieved 99.99%, and extending its farthest-distance bias tie from two distances to three removes one parameter while preserving embedding rank, MLP width, attention gains, and normalization contrast.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9997, "parameters": 854, "training_steps": 4999}
prior_hypothesis: The resulting 854-parameter transformer will maintain at least 99% accuracy because it preserves the essential unique-versus-shared scale contrast while allowing the six-coordinate group, rather than the singleton, to learn its magnitude.

## Recent verification evidence

RECENT RESULT
hypothesis: The resulting 857-parameter transformer will maintain at least 99% accuracy because the otherwise identical 858-parameter model achieved 99.93%, and extending its successful final-LayerNorm scale sharing from three coordinates to four removes only one learned parameter.
change: Adopt the verified single-coefficient-anchored rank-six embedding and five-gain attention pattern, then share four adjacent final LayerNorm scales while preserving the six-unit MLP and all training and decoding behavior.
mechanism: Quadruple-shared final normalization scale
evidence_used: The 858-parameter triple-shared LayerNorm design achieved 99.93%; further compression of the embedding, attention gains, or MLP previously failed, so one additional normalization-scale tie is the smallest evidence-supported next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9968, "parameters": 857, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 856-parameter transformer will maintain at least 99% accuracy because the otherwise identical 857-parameter model achieved 99.68%, and extending its successful final-LayerNorm scale sharing from four coordinates to five removes only one learned parameter while leaving the sensitive rank-six embedding, attention gains, and six-unit MLP unchanged.
change: Share five adjacent final LayerNorm scale coordinates instead of four, reducing the learned scale vector by one parameter.
mechanism: Quintuple-shared final normalization scale
evidence_used: Successive pair-, triple-, and quadruple-shared final LayerNorm designs achieved 99.65%, 99.93%, and 99.68%, respectively; the current 857-parameter result therefore supports one more isolated normalization-scale tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 856, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 776-parameter transformer will maintain at least 99% accuracy because the linear rank-five codebook narrowly reached 98.50%, while a learned GELU lift can generate a full-rank six-feature embedding matrix from five per-token coordinates instead of permanently deleting the sixth feature direction.
change: Replace the anchored dense rank-six vocabulary matrix with five-dimensional learned token codes decoded through a shared learned 5×6 GELU map, retaining tied learned input embeddings and output logits.
mechanism: Nonlinear rank-restoring vocabulary manifold
evidence_used: Rank six repeatedly exceeded 99% while fixed linear rank five missed by only 0.5 percentage points; this tests whether the load-bearing property is six-dimensional decoded features rather than six independent coefficients per vocabulary item.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.38189999999999996, "parameters": 776, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 855-parameter transformer will maintain at least 99% accuracy because the otherwise identical 856-parameter quintuple-shared design achieved 99.94%, and extending that scale-sharing group by one coordinate removes only one additional learned parameter relative to the qualified design.
change: Share six adjacent final LayerNorm scale coordinates while retaining the fixed anchor, rank-six embedding, five-gain attention pattern, and six-unit MLP.
mechanism: Sextuple-shared final normalization scale
evidence_used: Successive pair-, triple-, quadruple-, and quintuple-shared final LayerNorm designs achieved 99.65%, 99.93%, 99.68%, and 99.94%; this motivates the next isolated normalization-scale tie while avoiding the failed embedding, attention-gain, and MLP compressions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976, "parameters": 855, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 854-parameter transformer will maintain at least 99% accuracy because the otherwise identical 855-parameter sextuple-shared design achieved 99.76%, and extending its shared final-LayerNorm scale group by one coordinate removes only one learned parameter.
change: Adopt the qualified anchored rank-six vocabulary embedding, five-gain attention pattern, and six-unit MLP, then share all seven trainable final-LayerNorm scale coordinates while retaining the fixed scale anchor.
mechanism: Septuple-shared final normalization scale
evidence_used: Successive triple-, quadruple-, quintuple-, and sextuple-shared LayerNorm designs achieved 99.93%, 99.68%, 99.94%, and 99.76%; another isolated normalization-scale tie is better supported than further compressing the embedding, attention gains, or MLP, which previously failed.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.10279999999999999, "parameters": 854, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 854-parameter transformer will maintain at least 99% accuracy because it preserves the unique-versus-shared scale contrast lost by the failed septuple-sharing design, while fixing the shared sextet to the existing unit anchor.
change: Replace the two learned final-LayerNorm scales `[a,b,b,b,b,b,b,1]` with one learned contrast scale `[a,1,1,1,1,1,1,1]`.
mechanism: Contrast-only final normalization scale
evidence_used: The 855-parameter sextuple-shared design achieved 99.76%, whereas tying its unique scale into the shared group collapsed accuracy to 10.28%; this suggests the unique coordinate contrast is essential, while the shared group’s absolute scale is the safer parameter to anchor.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9754999999999999, "parameters": 854, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 854-parameter transformer will maintain at least 99% accuracy because it preserves the essential unique-versus-shared scale contrast while allowing the six-coordinate group, rather than the singleton, to learn its magnitude.
change: Replace final LayerNorm scales `[a,b,c,c,c,c,c,1]` with `[1,b,b,b,b,b,b,1]`, reducing three learned scales to one.
mechanism: Endpoint-anchored shared normalization contrast
evidence_used: The 855-parameter `[a,b,b,b,b,b,b,1]` design achieved 99.76%; merging the shared sextet with the anchor reached 97.55%, while merging the singleton with the sextet collapsed, leaving this third pairwise class merge as the most informative untested 854-parameter design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 854, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 853-parameter transformer will maintain at least 99% accuracy because the qualified 854-parameter endpoint-anchored LayerNorm design achieved 99.97%, and merging two adjacent attention output-bias groups removes one parameter without reducing embedding rank, MLP width, or attention-gain capacity.
change: Adopt the qualified one-parameter final LayerNorm and reduce the attention output bias from `[a,b,b,c,c,c,c]` to `[a,b,b,b,b,b,b]`.
mechanism: Endpoint-anchored normalization with singleton-versus-sextet attention bias
evidence_used: The endpoint-anchored 854-parameter model reached 99.97%, while prior failures came from eliminating embedding coefficients, reducing attention gains, shrinking the MLP, or removing the final-scale contrast; this tests a distinct isolated one-parameter tie.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The 853-parameter transformer will achieve at least 99% accuracy because the qualified 854-parameter endpoint-anchored design achieved 99.97%, and this changes only one attention output-bias degree of freedom; the prior test of this architecture yielded no accuracy evidence because verification timed out.
change: Adopt the qualified anchored rank-six embedding, five-gain attention pattern, six-unit MLP, and endpoint-anchored final LayerNorm; merge the attention output bias into `[a,b,b,b,b,b,b]` and reduce validation frequency so the previously censored 853-parameter candidate can complete.
mechanism: Endpoint-anchored normalization with singleton-versus-sextet attention bias
evidence_used: The 854-parameter endpoint-anchored model reached 99.97%; the only attempted 853-parameter output-bias tie did not finish within the time limit, so completing that isolated one-parameter experiment is the most informative next test.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The 853-parameter transformer will achieve at least 99% accuracy because the qualified 854-parameter model reached 99.97%, while this removes only one attention output-bias degree of freedom; evaluating only after training will allow the previously time-censored architecture to complete.
change: Merge the attention output-bias groups from `[a,b,b,c,c,c,c]` to `[a,b,b,b,b,b,b]` and run validation only at the final step.
mechanism: Singleton-versus-sextet attention bias with final-only validation
evidence_used: The 854-parameter endpoint-anchored design achieved 99.97%. Both prior attempts at this isolated 853-parameter reduction timed out without accuracy evidence, including one with merely reduced validation frequency, motivating a definitive final-only evaluation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: The resulting 853-parameter transformer will maintain at least 99% accuracy because the qualified 854-parameter endpoint-anchored design achieved 99.97%, while tying only the two sparsest, farthest relative-distance biases preserves embedding rank, MLP width, attention gains, and all established scale contrasts.
change: Adopt the qualified endpoint-anchored final LayerNorm and reduce the far-relative-bias vector from eleven parameters to ten by sharing its final value across the two greatest causal distances.
mechanism: Farthest-distance relative-attention bias tie
evidence_used: The endpoint-anchored 854-parameter model reached 99.97%; three attempts to remove an attention output-bias degree of freedom produced only timeouts, so testing an isolated parameter tie in the least frequently used relative-distance boundary is the most informative distinct 853-parameter experiment.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 853, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 852-parameter transformer will maintain at least 99% accuracy because the qualified 853-parameter design achieved 99.99%, and extending its farthest-distance bias tie from two distances to three removes one parameter while preserving embedding rank, MLP width, attention gains, and normalization contrast.
change: Adopt the qualified endpoint-anchored final LayerNorm and parameterize the eleven far-distance biases with nine values by sharing one value across the three greatest causal distances.
mechanism: Three-way farthest-distance relative-attention bias tie
evidence_used: The 853-parameter model tying the two farthest relative-distance biases achieved 99.99%; extending that successful isolated boundary tie is the smallest evidence-supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 852, "training_steps": 4999}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
