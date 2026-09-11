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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.813546834047884, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20859080200195312, "validation_score": 9273.413704952223}
prior_hypothesis: Scaling evaluation logits by 1.2 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.209594.

REFERENCE DESIGN 1
verified_results: {"batch_size": 256, "examples_processed": 100000, "optimizer_steps": 392, "parameters": 105866, "training_seconds": 12.175030375132337, "validation_accuracy": 0.8928, "validation_correct": 8928, "validation_cross_entropy": 0.33585684967041013, "validation_score": 8928.37429160177}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.80794441699982, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177444458008, "validation_score": 9273.413729961598}
prior_hypothesis: Scaling evaluation logits by 1.17823 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517870.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 71.25856295810081, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177448272705, "validation_score": 9273.413729961467}
prior_hypothesis: Scaling evaluation logits by 1.17810 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517744.

## Recent verification evidence

RECENT RESULT
hypothesis: The best verified pairwise-refinement model with a 10× lower terminal learning rate will exceed 9,273 correct predictions by allowing more precise late-stage convergence.
change: Restore the verified batch-64, groups-64 5×5 architecture and reduce the cosine schedule’s minimum learning rate from 1e-4 to 1e-5.
mechanism: Pairwise 5×5 refinement with deeper cosine annealing
evidence_used: Reference Design 3 achieved the best result at 9,273 correct, while update densification improved cross-entropy; its 0.2151 cross-entropy leaves room for a lower terminal rate to refine the final solution.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.11338229198009, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.21639883575439453, "validation_score": 9270.411049390466}

RECENT RESULT
hypothesis: Evaluating a 0.99-decay moving average of the best model’s training trajectory will exceed 9,273 correct predictions by reducing noise from its 1,564 small-batch updates without altering its successful architecture or learning-rate schedule.
change: Preserve the current model and training procedure, maintain an exponential moving average of every learned parameter after optimizer steps, and install those averaged parameters after the final step for validation.
mechanism: Exponential moving-average checkpoint smoothing
evidence_used: The current pairwise-refinement model is best at 9,273 correct, while residual refinement and a 10× lower terminal learning rate fell to 9,268 and 9,270; this motivates smoothing the verified trajectory rather than changing its features or endpoint schedule again.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 44.63960437499918, "validation_accuracy": 0.9269, "validation_correct": 9269, "validation_cross_entropy": 0.2148154499053955, "validation_score": 9269.411585150681}

RECENT RESULT
hypothesis: Replacing the single pairwise 5×5 refinement with two pairwise 3×3 stages will exceed 9,273 correct predictions by preserving its effective 5×5 receptive field while adding an intermediate nonlinearity.
change: Restore the best verified 1e-4 terminal learning rate and factor the 5×5 groups-64 refinement into two batch-normalized 3×3 groups-64 stages, reducing parameters to approximately 246,890.
mechanism: Two-stage nonlinear pairwise refinement
evidence_used: Single-stage pairwise refinement achieved the best result at 9,273 correct, while stronger four-channel coupling fell to 9,244 and residual refinement reached 9,268; this motivates retaining non-residual pairwise coupling while testing additional nonlinear depth.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 246890, "training_seconds": 59.85714500001632, "validation_accuracy": 0.9245, "validation_correct": 9245, "validation_cross_entropy": 0.21580959205627442, "validation_score": 9245.411248606088}

RECENT RESULT
hypothesis: Restoring the best non-residual pairwise 5×5 refinement and replacing its final non-overlapping pool with a 3×3 stride-2 pool will exceed 9,273 correct predictions by retaining information from all 7×7 feature positions.
change: Replace the residual refinement with the verified plain groups-64 convolution and use overlapping final max pooling, while preserving the optimizer, loss, schedule, batch size, and 248,426-parameter count.
mechanism: Full-coverage overlapping spatial pooling
evidence_used: Plain pairwise refinement achieved the best 9,273 correct, while residual refinement fell to 9,268 and two-stage refinement to 9,245. Its final 2×2 stride-2 pool maps 7×7 to 3×3 while ignoring the last row and column; 3×3 stride-2 pooling covers the full map without adding parameters.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 46.623481875052676, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.21919828262329102, "validation_score": 9259.410105564555}

RECENT RESULT
hypothesis: Decaying label smoothing from 0.02 to zero will exceed 9,273 correct predictions by retaining early regularization while allowing hard-label margin refinement late in training.
change: Preserve the best architecture, optimizer, schedule, and batch size; replace constant label smoothing with cosine-decayed smoothing.
mechanism: Annealed target sharpening
evidence_used: The current design is best at 9,273 correct, while residual refinement, deeper cosine annealing, EMA, and alternate pooling all regressed, motivating an isolated loss-target change instead of another architectural or learning-rate modification.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 48.364963749889284, "validation_accuracy": 0.9244, "validation_correct": 9244, "validation_cross_entropy": 0.20777745056152344, "validation_score": 9244.41398355282}

RECENT RESULT
hypothesis: Removing the classifier’s mild dropout while restoring the verified 1e-4 terminal learning rate will exceed 9,273 correct predictions by improving limited-exposure feature utilization while retaining label smoothing, weight decay, and batch normalization as regularizers.
change: Restore the best verified cosine minimum learning rate and replace the classifier dropout with an identity operation, preserving architecture dimensions, parameter count, batch size, optimizer, and loss.
mechanism: Dropout-free classifier with restored best cosine endpoint
evidence_used: The plain pairwise 5×5 model achieved the best result of 9,273 correct with a 1e-4 terminal rate; lowering it to 1e-5 fell to 9,270, and annealing label smoothing fell to 9,244. This motivates restoring both verified settings and testing the previously unchanged classifier dropout in isolation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.65080608311109, "validation_accuracy": 0.9264, "validation_correct": 9264, "validation_cross_entropy": 0.21516940689086914, "validation_score": 9264.411465263333}

RECENT RESULT
hypothesis: Restoring the best plain pairwise-refinement model and multiplying validation logits by 1.1 will preserve its 9,273 correct predictions while reducing cross-entropy below 0.2151.
change: Remove the unsuccessful residual identity path, restore non-residual 5×5 groups-64 refinement, and apply fixed positive logit scaling only in evaluation mode.
mechanism: Evaluation-only logit sharpening
evidence_used: Reference Design 3 achieved the best 9,273 correct predictions; annealed target sharpening lowered cross-entropy to 0.2078 but changed decisions. Evaluation-only scaling sharpens confidence without changing argmax predictions.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 55.08498666691594, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20959415397644043, "validation_score": 9273.413361786146}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.2 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.209594.
change: Increase the evaluation-only positive logit multiplier from 1.1 to 1.2 without changing training.
mechanism: Stronger evaluation-only logit sharpening
evidence_used: Scaling logits by 1.1 preserved the best 9,273 correct predictions and reduced cross-entropy from 0.215099 to 0.209594, motivating one further sharpening step.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 54.813546834047884, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20859080200195312, "validation_score": 9273.413704952223}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208591.
change: Apply a fixed 1.17 multiplier to logits only during evaluation, leaving training and learned parameters unchanged.
mechanism: Evaluation-only temperature calibration
evidence_used: Multipliers of 1.1 and 1.2 preserved 9,273 correct while reducing cross-entropy from 0.215099 to 0.209594 and 0.208591; these measurements suggest the calibration optimum lies near 1.17.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.78242424991913, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20852822303771973, "validation_score": 9273.413726374336}

RECENT RESULT
hypothesis: Restoring the verified 1e-4 cosine endpoint and scaling evaluation logits by 1.179 will preserve 9,273 correct predictions while reducing cross-entropy below 0.208528.
change: Restore the best training schedule and apply a fixed 1.179 logit multiplier only during evaluation.
mechanism: Quadratically refined evaluation-only temperature calibration
evidence_used: Evaluation scales 1.1, 1.17, and 1.2 preserved 9,273 correct with cross-entropies 0.209594, 0.208528, and 0.208591; quadratic interpolation places the minimum near 1.179.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 56.5536045411136, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20851787033081054, "validation_score": 9273.413729918502}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17823 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517870.
change: Refine the evaluation-only logit multiplier from 1.179 to the quadratic-fit optimum of approximately 1.17823.
mechanism: Fine-grained evaluation temperature calibration
evidence_used: Scales 1.17, 1.179, and 1.2 preserved 9,273 correct with cross-entropies 0.208528223, 0.208517870, and 0.208590802; quadratic interpolation of these measurements places the minimum near 1.17823.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 62.80794441699982, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177444458008, "validation_score": 9273.413729961598}

RECENT RESULT
hypothesis: Scaling evaluation logits by 1.17810 will preserve all 9,273 argmax predictions while reducing validation cross-entropy below 0.208517744.
change: Refine the evaluation-only positive logit multiplier from 1.17 to 1.17810 without changing training or learned parameters.
mechanism: Fine-grained evaluation temperature calibration
evidence_used: Scales 1.17, 1.17823, and 1.179 preserved 9,273 correct with cross-entropies 0.208528223, 0.208517744, and 0.208517870; local quadratic interpolation places the minimum near 1.17810.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 248426, "training_seconds": 71.25856295810081, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.2085177448272705, "validation_score": 9273.413729961467}



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
