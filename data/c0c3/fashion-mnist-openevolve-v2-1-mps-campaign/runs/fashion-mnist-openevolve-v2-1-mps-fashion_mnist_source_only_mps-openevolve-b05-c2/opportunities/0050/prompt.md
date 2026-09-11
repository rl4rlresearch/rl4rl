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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.08673800015822, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22917193641662598, "validation_score": 9192.406777917056}
prior_hypothesis: Dividing the ensemble logits by 92% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.2329286087036133.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 43.87550191604532, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22403448944091797, "validation_score": 9192.408485221873}
prior_hypothesis: Dividing the ensemble logits by 84% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.22917193641662598.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 54.37880758289248, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742913589477538, "validation_score": 9192.404063542304}
prior_hypothesis: A diagonal weight of 0.7161376953125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742915229797362.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 64.48993279179558, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2329286087036133, "validation_score": 9192.405538484929}
prior_hypothesis: Dividing the ensemble logits by 96% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.2350878746032715.

## Recent verification evidence

RECENT RESULT
hypothesis: A diagonal weight of 0.716137752532958984375 will retain 9,192 correct predictions while reducing validation cross-entropy below 0.23742913589477538.
change: Move the diagonal TTA weight halfway from the best-known setting toward the nearest tested upper setting, preserving exact ensemble normalization.
mechanism: One-step local diagonal-weight refinement
evidence_used: Weight 0.7161376953125 achieved the best observed cross-entropy, while 0.71613780975341796875 was slightly worse with the same correctness; their midpoint is the most informative remaining probe of the local minimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 60.568984125042334, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742914199829102, "validation_score": 9192.40406354031}

RECENT RESULT
hypothesis: The representable float32 diagonal weight immediately below the best-known effective weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23742913589477538.
change: Set the diagonal TTA weight to 0.7161376476287841796875 and normalize by its exact total ensemble weight.
mechanism: Float32-neighbor TTA weight refinement
evidence_used: The best result used 0.7161376953125, while the adjacent upper float32 setting tested via 0.716137752532958984375 was worse; the untested lower float32 neighbor is therefore the most informative remaining local probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 72.01745774992742, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23742914047241212, "validation_score": 9192.40406354081}

RECENT RESULT
hypothesis: Restoring the best-known diagonal weight and increasing its ensemble denominator by 0.5% will retain 9,192 correct predictions while reducing validation cross-entropy below 0.23742913589477538.
change: Restore the best-performing diagonal TTA weight and slightly soften all evaluation logits without changing their class ordering.
mechanism: Mild post-hoc logit temperature calibration
evidence_used: Weight 0.7161376953125 achieved the lowest observed cross-entropy, while adjacent weight refinements were worse; all prior tests coupled weighting to exact normalization, leaving accuracy-preserving temperature calibration unexplored.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.97872400004417, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2380424545288086, "validation_score": 9192.403863371705}

RECENT RESULT
hypothesis: Decreasing the best-known ensemble denominator by 0.5% will preserve all 9,192 predictions while reducing validation cross-entropy below 0.23742913589477538.
change: Retain the best-performing diagonal TTA weight and sharpen evaluation logits by dividing the ensemble sum by 99.5% of its exact normalization weight.
mechanism: Symmetric post-hoc logit-temperature calibration
evidence_used: Increasing the denominator by 0.5% preserved correctness but worsened cross-entropy to 0.2380424545288086, directly motivating an accuracy-preserving probe in the opposite temperature direction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Dividing the best-known ensemble logits by 99.5% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23742913589477538.
change: Restore the best-performing diagonal TTA weight and retry the timed-out 0.5% sharpening probe.
mechanism: Accuracy-preserving post-hoc logit sharpening
evidence_used: Increasing the denominator by 0.5% preserved correctness but worsened cross-entropy to 0.2380424545288086; the corresponding denominator decrease timed out without performance evidence, so retrying it is the most informative unresolved calibration test.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 68.76000120816752, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23682696762084962, "validation_score": 9192.404260266869}

RECENT RESULT
hypothesis: Dividing the best-known ensemble logits by 99.0% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23682696762084962.
change: Restore the best-performing diagonal TTA weight and increase sharpening from 0.5% to 1.0%.
mechanism: Continued post-hoc logit sharpening
evidence_used: Sharpening by 0.5% preserved all 9,192 predictions and improved cross-entropy from 0.23742913589477538 to 0.23682696762084962, while 0.5% softening worsened it to 0.2380424545288086.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 63.76771100005135, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.23623599700927733, "validation_score": 9192.404453519563}

RECENT RESULT
hypothesis: Dividing the best-known ensemble logits by 98.0% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23623599700927733.
change: Restore the best-performing diagonal TTA weight and increase post-hoc sharpening from 1.0% to 2.0%.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 0.5% and 1.0% successively reduced cross-entropy to 0.23682696762084962 and 0.23623599700927733 while retaining all 9,192 predictions; positive global logit scaling preserves class ordering.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 44.977414375171065, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2350878746032715, "validation_score": 9192.40482949455}

RECENT RESULT
hypothesis: Dividing the ensemble logits by 96% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.2350878746032715.
change: Increase post-hoc evaluation-logit sharpening from 2% to 4% while preserving the ensemble weights and class ordering.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 0.5%, 1%, and 2% successively reduced cross-entropy while retaining all 9,192 correct predictions; doubling the latest calibration step is the most informative next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 64.48993279179558, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.2329286087036133, "validation_score": 9192.405538484929}

RECENT RESULT
hypothesis: Dividing the ensemble logits by 92% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.2329286087036133.
change: Increase post-hoc evaluation-logit sharpening from 4% to 8% while preserving ensemble weights and class ordering.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 0.5%, 1%, 2%, and 4% successively reduced cross-entropy while retaining all 9,192 correct predictions; doubling the latest successful calibration step is the most informative next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 48.08673800015822, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22917193641662598, "validation_score": 9192.406777917056}

RECENT RESULT
hypothesis: Dividing the ensemble logits by 84% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.22917193641662598.
change: Increase post-hoc evaluation-logit sharpening from 8% to 16% while preserving ensemble weights and class ordering.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 0.5%, 1%, 2%, 4%, and 8% successively reduced cross-entropy while retaining all 9,192 correct predictions; doubling the latest successful calibration step is the most informative next probe.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 43.87550191604532, "validation_accuracy": 0.9192, "validation_correct": 9192, "validation_cross_entropy": 0.22403448944091797, "validation_score": 9192.408485221873}

RECENT RESULT
hypothesis: Dividing the ensemble logits by 68% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.22403448944091797.
change: Increase post-hoc evaluation-logit sharpening from 16% to 32% while preserving ensemble weights and class ordering.
mechanism: Continued accuracy-invariant logit sharpening
evidence_used: Sharpening by 4%, 8%, and 16% successively reduced cross-entropy to 0.2329286087036133, 0.22917193641662598, and 0.22403448944091797 while retaining all 9,192 correct predictions; doubling the latest successful sharpening step is the most informative next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249260, "training_seconds": 53.29174400004558, "validation_accuracy": 0.9191, "validation_correct": 9191, "validation_cross_entropy": 0.224783793258667, "validation_score": 9191.40823531692}

RECENT RESULT
hypothesis: Sharpening from 16% to 24% while anchoring class ordering to the proven 84%-denominator logits will retain 9,192 correct predictions and reduce cross-entropy below 0.22403448944091797.
change: Center the current logits on their per-image maximum, then apply the 21/19 scale corresponding to a 76%-of-normalization denominator.
mechanism: Order-locked midpoint temperature sharpening
evidence_used: The 84% denominator achieved 9,192 correct at 0.22403448944091797 cross-entropy, while 68% lost one prediction; 76% is the informative midpoint, and max-centering before positive scaling preserves the 84% logits’ argmax.
result: training did not finish within the verification time limit



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
