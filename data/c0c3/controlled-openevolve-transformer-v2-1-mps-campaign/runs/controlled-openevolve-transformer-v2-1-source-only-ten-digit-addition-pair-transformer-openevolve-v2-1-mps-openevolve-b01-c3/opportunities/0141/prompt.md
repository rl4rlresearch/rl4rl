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
verified_results: {"accuracy": 0.9965, "parameters": 859, "training_steps": 4999}
prior_hypothesis: The resulting 859-parameter transformer will maintain at least 99% accuracy because the 860-parameter model achieved 99.86%, and sharing two adjacent final LayerNorm scales removes one parameter without further constraining the embedding codebook whose second anchored coefficient caused the prior 859-parameter model to collapse to 64.24%.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1004, "training_steps": 4999}
prior_hypothesis: A 1,004-parameter transformer will maintain at least 99% accuracy because successive MLP-width reductions to eleven, ten, and nine units achieved 100%, 99.89%, and 99.96%, respectively; reducing the verified nine-unit design by one more unit tests the same isolated 14-parameter reduction.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9993000000000001, "parameters": 858, "training_steps": 4999}
prior_hypothesis: The resulting 858-parameter transformer will maintain at least 99% accuracy because the 859-parameter model achieved 99.65%, and extending its successful adjacent final-LayerNorm scale tie by one coordinate removes a single parameter without further constraining the sensitive rank-six vocabulary codebook.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9994, "parameters": 856, "training_steps": 4999}
prior_hypothesis: The resulting 856-parameter transformer will maintain at least 99% accuracy because the otherwise identical 857-parameter model achieved 99.68%, and extending its successful final-LayerNorm scale sharing from four coordinates to five removes only one learned parameter while leaving the sensitive rank-six embedding, attention gains, and six-unit MLP unchanged.

## Recent verification evidence

RECENT RESULT
hypothesis: The resulting 973-parameter transformer will maintain at least 99% accuracy because the current 974-parameter model achieved 99.97%, while the same `[a,b,b,c,c,d,d]` gain pattern achieved 99.88% in the larger-MLP design.
change: Replace the five learned secondary-head gains expanded as `[a,b,c,d,d,e,e]` with four gains expanded as `[a,b,b,c,c,d,d]`, removing one learned parameter while retaining the verified six-unit MLP.
mechanism: Six-unit MLP with three-pair secondary-head gain sharing
evidence_used: The current six-unit, two-pair-gain model reached 99.97%, and the independently verified three-pair pattern reached 99.88%; this supports testing one isolated additional adjacent gain tie without crossing the failed five-unit MLP bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 973, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 864-parameter transformer will maintain at least 99% accuracy because the six-unit MLP reached 100% while the 973-parameter gain-tied model reached 99.93%; retaining that nonlinear width but constraining the previously untested token/logit codebook to six centered features should preserve the learned computation.
change: Replace the full seven-coordinate centered vocabulary embedding with a learned six-coordinate embedding projected through a fixed orthonormal mean-zero basis, removing 111 parameters while preserving the eight-dimensional residual stream, six-unit MLP, causal attention, and tied learned logits.
mechanism: Rank-six tied vocabulary bottleneck
evidence_used: The five-unit MLP failed at 96.48%, identifying nonlinear width as load-bearing, whereas six units repeatedly exceeded 99%. The old designs implicitly assumed all seven learned vocabulary-feature coordinates were necessary; this patch instead tests a lower-rank learned input/output representation without revisiting the failed MLP bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 862, "training_steps": 4999}

RECENT RESULT
hypothesis: A 749-parameter transformer will maintain at least 99% accuracy because the verified rank-six, 862-parameter design achieved 99.99%; reducing only its learned vocabulary feature rank by one tests whether that high-margin representation retains another 113-parameter compression.
change: Adopt the verified final-pair gain sharing and reduce the centered tied vocabulary embedding from six to five learned feature coordinates, preserving the six-unit MLP, eight-dimensional residual stream, causal attention, training procedure, and protected decoding interface.
mechanism: Rank-five tied vocabulary bottleneck
evidence_used: The rank-six vocabulary bottleneck reached 99.99% at 862 parameters, while the five-unit MLP failed at 96.48%; this motivates compressing the successful vocabulary codebook rather than the demonstrated load-bearing nonlinear width.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.985, "parameters": 749, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 860-parameter transformer will maintain at least 99% accuracy because the rank-six 862-parameter design achieved 99.99%, while the `[a,b,b,c,c,d,d]` gain pattern independently achieved 99.93% with the same six-unit MLP.
change: Adopt the verified rank-six tied vocabulary embedding and six-unit MLP, then represent seven secondary-head gains with four learned values expanded as `[a,b,b,c,c,d,d]`.
mechanism: Rank-six vocabulary bottleneck with three-pair secondary-head gain sharing
evidence_used: Rank five failed at 98.50%, so this preserves the successful rank-six embedding; the rank-six final-pair-tied model reached 99.99%, and three-pair gain sharing already exceeded the accuracy threshold in the 973-parameter design.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7854000000000001, "parameters": 860, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 861-parameter transformer will maintain at least 99% accuracy because the current rank-six, final-pair-tied model achieved 99.99%, while the same `[a,b,c,d,d,e,e]` gain pattern achieved 99.97% with the six-unit MLP before vocabulary compression.
change: Replace six learned secondary-head gains expanded as `[a,b,c,d,e,f,f]` with five gains expanded as `[a,b,c,d,d,e,e]`, removing one parameter while preserving the successful rank-six embedding and six-unit MLP.
mechanism: Rank-six vocabulary bottleneck with two-pair secondary-head gain sharing
evidence_used: The 862-parameter rank-six model reached 99.99%; although jumping directly to four gains failed at 78.54%, the five-gain pattern independently retained 99.97%, making this isolated intermediate tie the most informative next compression.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 861, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 860-parameter transformer will maintain at least 99% accuracy because it preserves the successful rank-six embedding while fixing only one of its 678 latent coefficients, rather than removing the entire 113-coefficient feature column that reduced accuracy to 98.50%.
change: Store one fewer learned vocabulary-codebook coefficient and append a fixed zero when reconstructing the unchanged rank-six latent matrix.
mechanism: Single-coefficient anchored rank-six vocabulary codebook
evidence_used: The current 861-parameter rank-six model achieved 99.99% accuracy, whereas reducing the whole embedding to rank five removed 113 parameters and narrowly failed; this isolates the smallest possible embedding compression without reducing its rank.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 860, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 859-parameter transformer will maintain at least 99% accuracy because the otherwise identical 860-parameter model achieved 99.86% after anchoring one of 678 rank-six codebook coefficients; anchoring one additional coefficient is the smallest possible follow-up compression and preserves the successful rank-six representation.
change: Replace the full-rank vocabulary embedding with a rank-six latent embedding containing two fixed-zero coefficients, and use the verified five-gain expansion `[a,b,c,d,d,e,e]`.
mechanism: Two-coefficient anchored rank-six vocabulary codebook
evidence_used: The 860-parameter single-coefficient-anchored rank-six model achieved 99.86%, whereas removing an entire 113-coefficient feature column failed at 98.50%; this motivates another isolated coefficient anchor without reducing rank.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6424, "parameters": 859, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 859-parameter transformer will maintain at least 99% accuracy because the 860-parameter model achieved 99.86%, and sharing two adjacent final LayerNorm scales removes one parameter without further constraining the embedding codebook whose second anchored coefficient caused the prior 859-parameter model to collapse to 64.24%.
change: Tie the final two trainable LayerNorm scale coordinates while retaining the existing fixed scale anchor, rank-six embedding, five-gain attention pattern, and six-unit MLP.
mechanism: Pair-shared final normalization scale
evidence_used: The single-coefficient-anchored 860-parameter design met the requirement at 99.86%, whereas anchoring a second embedding coefficient failed severely; this motivates testing the same one-parameter reduction in a different, localized component.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9965, "parameters": 859, "training_steps": 4999}

RECENT RESULT
hypothesis: The resulting 858-parameter transformer will maintain at least 99% accuracy because the 859-parameter model achieved 99.65%, and extending its successful adjacent final-LayerNorm scale tie by one coordinate removes a single parameter without further constraining the sensitive rank-six vocabulary codebook.
change: Adopt the verified single-coefficient-anchored rank-six embedding and five-gain attention pattern, then share three adjacent final LayerNorm scales instead of two, producing an 858-parameter model.
mechanism: Triple-shared final normalization scale
evidence_used: The 859-parameter pair-shared LayerNorm design achieved 99.65%, whereas anchoring a second embedding coefficient collapsed accuracy to 64.24%; this motivates applying the next isolated one-parameter reduction to normalization rather than the embedding.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 858, "training_steps": 4999}

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
