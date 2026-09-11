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
verified_results: {"accuracy": 0.9911, "parameters": 1629, "training_steps": 4999}
prior_hypothesis: Removing the softmax-invisible coordinate from the third key row of the second head will reduce the model from 1630 to 1629 parameters while retaining at least 99% accuracy, because two removals in that head succeeded whereas the second removal in the first head failed.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the LayerNorm-invariant common-shift coordinate from the attention output bias will produce a 1634-parameter model with at least 99% accuracy when virtual eight-coordinate AdamW preserves the successful optimizer dynamics.
change: Store seven output-bias coordinates with the eighth fixed at zero, and optimize them using virtual eight-dimensional AdamW moments and gauge-aware gradient clipping.
mechanism: Quotient-aware AdamW for a gauge-fixed attention output bias
evidence_used: The 1635-parameter model reached 99.32%, while ordinary fixed-coordinate and orthonormal output-bias gauges reached only 73.43% and 4.04%; this suggests the exact redundancy is removable but AdamW’s coordinate-dependent dynamics must be preserved.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one LayerNorm-invariant coordinate from the final positional vector will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy when virtual eight-coordinate AdamW preserves the full embedding’s initialization and optimizer dynamics.
change: Represent the final positional vector with seven learned coordinate differences and one fixed zero, initialize it as the exact gauge-equivalent of an eight-coordinate draw, and optimize it with the existing gauge-aware AdamW and gradient clipping.
mechanism: Quotient-aware positional-embedding gauge fixing
evidence_used: Ordinary positional gauge fixing collapsed to 17.26%, but the analogous output-bias gauge recovered from 73.43% to 99.85% when virtual full-coordinate AdamW was introduced; this directly motivates applying the successful optimizer treatment to the exact positional gauge.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6977, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the LayerNorm-invariant common-shift coordinate from the MLP output bias will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy when virtual eight-coordinate AdamW preserves its optimizer dynamics.
change: Replace the eight-coordinate MLP output bias with seven learned coordinates and one fixed zero, then include it in the existing gauge-aware optimizer and gradient clipping.
mechanism: Quotient-aware MLP output-bias gauge fixing
evidence_used: Virtual full-coordinate AdamW made the analogous attention output-bias reduction reach 99.85% at 1634 parameters; unlike the failed positional gauge, the MLP bias is another zero-initialized output offset whose common shift passes through the residual stream and is canceled by downstream LayerNorms.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0882, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the global common-shift degree of freedom from the tied token embedding will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy, because the shift is canceled at transformer inputs by LayerNorm and adds only a vocabulary-wide constant to output logits, while virtual full-coordinate AdamW preserves the successful optimization dynamics.
change: Store all but one coordinate of the tied token embedding, reconstruct its final coordinate as zero, preserve gauge-equivalent initialization and RNG ordering, and optimize it alongside the attention output bias with virtual full-coordinate moments and clipping.
mechanism: Quotient-aware tied-embedding gauge fixing
evidence_used: The 1634-parameter attention-bias quotient reached 99.85% only with virtual full-coordinate AdamW; this patch applies that successful optimizer treatment to a distinct exact gauge while leaving all optimization-sensitive query, value, normalization, and MLP parameters intact.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9972, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the single global common-shift degree of freedom from the positional embedding will reduce the model from 1633 to 1632 parameters while retaining at least 99% accuracy, because this shift is canceled by LayerNorm and the successful virtual full-coordinate AdamW treatment preserves the original embedding’s optimization dynamics.
change: Represent the entire positional embedding with `GaugeFixedEmbedding` and optimize it alongside the tied token embedding and attention output bias using virtual full-coordinate moments and clipping.
mechanism: Quotient-aware global positional-embedding gauge fixing
evidence_used: The analogous global gauge removal from the tied token embedding achieved 99.72% at 1633 parameters. Unlike the failed final-row positional gauge at 69.77%, this applies the exact whole-embedding quotient structure already verified successfully.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6831, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final token-embedding row to zero will reduce the model from 1633 to 1626 parameters while retaining at least 99% accuracy, because any removed eight-dimensional row can be transferred to every positional vector without changing transformer inputs, while it changes output logits only by a vocabulary-wide constant; coupled virtual AdamW preserves the full-coordinate optimization dynamics.
change: Replace the single-scalar tied-embedding gauge with an eight-coordinate row gauge, transfer its initialization anchor into the positional embedding, and jointly optimize token and positional embeddings using reconstructed full gradients, moments, weight decay, and clipping.
mechanism: Coupled tied-token/position row-gauge quotient
evidence_used: The quotient-aware attention-bias model achieved 99.85% at 1634 parameters and the tied-embedding scalar quotient achieved 99.72% at 1633, showing virtual full-coordinate AdamW can preserve accuracy across exact gauges. The independent positional gauge failed at 68.31%, motivating this coupled gauge, which leaves the summed token-plus-position input unchanged rather than quotienting position embeddings alone.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5386, "parameters": 1626, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one additional feature coordinate of the final token row and transferring its column-wide offset into positional embeddings will reduce parameters from 1633 to 1632 while retaining at least 99% accuracy when coupled virtual-coordinate AdamW preserves full token and position gradients, moments, decay, and clipping.
change: Extend the successful tied-embedding scalar gauge by one token–position translation coordinate, reconstruct both omitted token coordinates, transfer the initialization anchor into positional embeddings, and jointly optimize the coupled embeddings in virtual full coordinates.
mechanism: One-coordinate coupled token–position translation quotient
evidence_used: The single tied-embedding gauge achieved 99.72% at 1633 parameters, whereas removing seven additional row-gauge coordinates at once reached only 53.86%; a one-coordinate titration is the smallest informative continuation while preserving the quotient-aware optimization treatment.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1431, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one key-projection coordinate will reduce the model from 1633 to 1632 parameters while retaining at least 99% accuracy, because LayerNorm confines token variation to a mean-zero subspace, making one inverse-scale key direction position-constant and therefore exactly invisible to causal softmax; virtual full-coordinate AdamW preserves the omitted coordinate’s optimizer dynamics.
change: Replace the QKV projection with a one-scalar gauge-fixed projection, preserve gauge-equivalent initialization and RNG ordering, and optimize its reconstructed full gradient with gauge-aware AdamW and clipping.
mechanism: Quotient-aware LayerNorm key-projection gauge fixing
evidence_used: The 1633-parameter tied-embedding quotient achieved 99.72%, while positional and MLP-bias quotients failed despite optimizer corrections; this motivates testing a distinct exact redundancy located inside the key projection, whose removed direction is directly canceled by softmax.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one additional softmax-invisible key-projection coordinate will reduce the model from 1632 to 1631 parameters while retaining at least 99% accuracy, because each key output row has an independent LayerNorm-induced constant-shift gauge and virtual full-coordinate AdamW preserves its optimizer dynamics.
change: Gauge-fix the final input coordinate of a second key-projection row, reconstruct both omitted gradients and optimizer coordinates, and include both virtual coordinates in gradient clipping.
mechanism: Second quotient-aware LayerNorm key-projection gauge
evidence_used: The first quotient-aware key-coordinate removal achieved 99.92% accuracy at 1632 parameters; extending that successful mechanism by exactly one coordinate is the smallest informative titration.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8759999999999999, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the softmax-invisible coordinate from the first key row of the second attention head will reduce the model from 1632 to 1631 parameters while retaining at least 99% accuracy, because it distributes the two exact key gauges across heads while preserving both omitted coordinates’ full-space AdamW dynamics.
change: Extend `GaugeFixedQKV` to omit the final input coordinate from the first key row of each head, and generalize virtual-gradient reconstruction, AdamW updates, and clipping to both coordinates.
mechanism: Cross-head quotient-aware key-projection gauge fixing
evidence_used: The first quotient-aware key-coordinate removal achieved 99.92% at 1632 parameters, whereas removing the adjacent second key row reached 87.60%; with four coordinates per head, that adjacent row shares the first head, motivating the smallest alternative titration in the other head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1631, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the softmax-invisible coordinate from the second key row of the second head will reduce the model from 1631 to 1630 parameters while retaining at least 99% accuracy; unlike the failed two-row first-head configuration, this tests whether within-head gauge sensitivity is head-specific while preserving the successful cross-head gauges.
change: Add the second key row of the second head to `GaugeFixedQKV.fixed_rows` and generalize virtual AdamW state allocation to the number of omitted coordinates.
mechanism: Third quotient-aware key gauge localized to the second attention head
evidence_used: Removing the first key row from each head achieved 99.93% at 1631 parameters, whereas removing the first two key rows from the first head achieved only 87.60%; placing the necessary third gauge in the second head is the closest untested titration.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1630, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the softmax-invisible coordinate from the third key row of the second head will reduce the model from 1630 to 1629 parameters while retaining at least 99% accuracy, because two removals in that head succeeded whereas the second removal in the first head failed.
change: Add the third key row of the second attention head to `GaugeFixedQKV.fixed_rows`; the existing generalized virtual AdamW and gradient-clipping logic will handle the fourth omitted coordinate.
mechanism: Fourth quotient-aware key gauge within the empirically robust second attention head
evidence_used: The 1630-parameter model reached 99.91% after removing two coordinates from the second head, while placing the second within-head removal in the first head reached only 87.60%; this motivates continuing the smallest one-coordinate titration in the empirically successful head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9911, "parameters": 1629, "training_steps": 4999}



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
