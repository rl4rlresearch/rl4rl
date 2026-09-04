# Improve fixed-exposure image classification

You are an autonomous ML engineer improving a learned classifier for 28×28
grayscale images in ten classes.

## Goal

Maximize `validation_score`. It ranks implementations first by the exact number
of correct predictions on the fixed 10,000-image validation set, then—only when
correct counts tie—by lower validation cross-entropy. Every verification starts
from a fresh initialization and presents exactly 100,000 examples from the
fixed 50,000-image training split.

You may change the model architecture, optimizer, loss, augmentation, batch
size, gradient handling, schedule, and other contents of `train.py`. The fixed
data split, normalization, example accounting, validation calculation,
250,000-learned-parameter ceiling, and device are not editable. The protected
loop calls the functions already defined in `train.py`; keep that interface
intact. The model must return one ten-class logit vector per image.

## Work boundaries

Maximize validation_score. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: validation_score, validation_correct, validation_accuracy, validation_cross_entropy, parameters, examples_processed, optimizer_steps, training_seconds, batch_size.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, external datasets, pretrained weights, or any
surrounding repository. Do not run training or validation yourself and do not
generate hidden alternatives. Return one patch for one implementation;
verification happens after you finish.

## Available designs

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 70.92235383414663, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799225845336915, "validation_score": 9322.417364967489}
prior_hypothesis: Raising the unanimous-correction translation contribution to 37.421875% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197992462.

## Recent verification evidence

RECENT RESULT
hypothesis: Translating 25% of training images by one reflected pixel will improve spatial robustness and produce more than 9,319 correct validation predictions.
change: Randomly replace one quarter of each training batch with a uniformly selected one-pixel translation matching the four evaluation-time views.
mechanism: Sparse one-pixel translation augmentation
evidence_used: Evaluation-time translated logits consistently reduced cross-entropy while preserving all 9,319 predictions; with the blend now locally saturated at 30%, training on those same perturbations is the most direct test of whether their complementary signal can improve correctness.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Allowing the 30% blend to change predictions only when all four translated views agree with its alternative class will exceed 9,319 correct predictions.
change: Retain the existing blend for argmax-preserving cases, while accepting prediction changes supported unanimously by all four translated views.
mechanism: Consensus-gated translation correction
evidence_used: Translation blending consistently reduced cross-entropy, indicating complementary signal; the current argmax gate prevents testing whether its strongest, view-consistent corrections improve accuracy.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 66.09472049982287, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.19801389999389649, "validation_score": 9321.417357427992}

RECENT RESULT
hypothesis: Allowing the 30% blend to change a base prediction when at least three translated views support the alternative will exceed 9,321 correct validation predictions.
change: Relax the evaluation-time correction gate from unanimous four-view agreement to agreement by at least three of four translated views.
mechanism: Three-of-four translation consensus correction
evidence_used: Unanimous translation consensus increased correctness from 9,319 to 9,321 while lowering cross-entropy to 0.198013900, showing that translated views can identify beneficial prediction changes; three-view consensus tests whether slightly broader agreement captures additional corrections.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 74.58839158294722, "validation_accuracy": 0.9319, "validation_correct": 9319, "validation_cross_entropy": 0.1979814826965332, "validation_score": 9319.417368721655}

RECENT RESULT
hypothesis: Accepting a changed prediction when at least three translated views support it and all four reject the base class will exceed 9,321 correct predictions.
change: Broaden unanimous correction only to three-of-four alternatives whose dissenting view also rejects the original prediction.
mechanism: Three-view consensus with unanimous base rejection
evidence_used: Unanimous consensus improved correctness from 9,319 to 9,321, while unrestricted three-of-four consensus fell to 9,319; requiring every view to reject the base class isolates a stronger subset of the added corrections.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Raising translation weight to 40% only when all four translated views unanimously support a changed class will retain the existing corrections and exceed 9,321 correct predictions.
change: Keep the argmax-preserving 30% blend, but use a 40% blend for prediction changes unanimously supported by all translated views.
mechanism: Unanimous-view confidence escalation
evidence_used: The 30% blend with unanimous correction improved correctness from 9,319 to 9,321, whereas unrestricted three-of-four agreement fell to 9,319; this tests additional corrections without weakening unanimity.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 35% translation blend used only for unanimous alternative-class corrections will preserve the 9,321-correct baseline and add at least one correct prediction.
change: Keep the argmax-preserving blend at 30%, but evaluate unanimous prediction changes with a separate 35% translation blend.
mechanism: Conservative unanimous-view confidence escalation
evidence_used: The 30% unanimous gate improved correctness from 9,319 to 9,321, while relaxing consensus to three views reduced it to 9,319; a 35% unanimous-only blend cautiously expands corrections without weakening the successful agreement criterion.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 68.91605966701172, "validation_accuracy": 0.9321, "validation_correct": 9321, "validation_cross_entropy": 0.1980042869567871, "validation_score": 9321.417360776955}

RECENT RESULT
hypothesis: Raising only the unanimous-correction translation blend from 35% to 37.5% will exceed 9,321 correct predictions while retaining the stricter four-view agreement gate.
change: Use a 37.5% translated-logit contribution for unanimous alternative-class corrections; keep the argmax-preserving blend at 30%.
mechanism: Intermediate unanimous-view confidence escalation
evidence_used: A 35% unanimous-only blend retained 9,321 correct and improved cross-entropy over the 30% correction blend, while relaxing consensus reduced correctness; 37.5% probes the unmeasured interval before the inconclusive timed-out 40% attempt.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 36.25% translation blend for unanimous alternative-class corrections will retain 9,321 correct predictions and reduce validation cross-entropy below 0.198004287.
change: Increase only the unanimous-correction translated-logit contribution from 35% to 36.25%, keeping the argmax-preserving blend at 30%.
mechanism: Midpoint unanimous-correction confidence escalation
evidence_used: Raising the unanimous-correction blend from 30% to 35% retained 9,321 correct and improved cross-entropy from 0.198013900 to 0.198004287; the 37.5% verification timed out, so 36.25% is the informative midpoint of the unmeasured interval.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 78.68217887496576, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799538192749022, "validation_score": 9322.417363879313}

RECENT RESULT
hypothesis: Raising the unanimous-correction translation blend to 36.875% will exceed 9,322 correct predictions while retaining four-view agreement.
change: Increase only the unanimous-correction translated-logit contribution from 36.25% to 36.875%; keep the argmax-preserving blend at 30%.
mechanism: Midpoint unanimous-correction confidence escalation
evidence_used: Increasing unanimous-correction blending from 35% to 36.25% raised validation correctness from 9,321 to 9,322; 36.875% is the informative midpoint between the best verified setting and the inconclusive 37.5% attempt.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 56.02167758299038, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979937026977539, "validation_score": 9322.417364464332}

RECENT RESULT
hypothesis: Raising the unanimous-correction translation blend to 37.1875% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197993703.
change: Increase only the unanimous-correction translated-logit contribution from 36.875% to 37.1875%, keeping the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction confidence escalation
evidence_used: Increasing the unanimous-correction blend from 36.25% to 36.875% retained 9,322 correct while lowering cross-entropy from 0.197995382 to 0.197993703; 37.1875% is the midpoint between the best verified setting and the inconclusive 37.5% attempt.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 62.557787999976426, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.1979928741455078, "validation_score": 9322.417364752988}

RECENT RESULT
hypothesis: Raising the unanimous-correction translation contribution to 37.34375% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197992874.
change: Increase only the unanimous-correction translated-logit contribution from 37.1875% to 37.34375%; keep the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction confidence escalation
evidence_used: Increasing the unanimous-correction blend from 36.875% to 37.1875% retained 9,322 correct while lowering cross-entropy from 0.197993703 to 0.197992874; 37.34375% is the next midpoint toward the inconclusive 37.5% setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 77.90597108285874, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799246215820313, "validation_score": 9322.417364896519}

RECENT RESULT
hypothesis: Raising the unanimous-correction translation contribution to 37.421875% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197992462.
change: Increase only the unanimous-correction translated-logit contribution from 37.34375% to 37.421875%; keep the argmax-preserving blend at 30%.
mechanism: Fine-grained unanimous-correction confidence escalation
evidence_used: Increasing the unanimous-correction blend from 37.1875% to 37.34375% retained 9,322 correct while lowering cross-entropy from 0.197992874 to 0.197992462; 37.421875% is the next midpoint toward the inconclusive 37.5% setting.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249934, "training_seconds": 70.92235383414663, "validation_accuracy": 0.9322, "validation_correct": 9322, "validation_cross_entropy": 0.19799225845336915, "validation_score": 9322.417364967489}



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
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.
