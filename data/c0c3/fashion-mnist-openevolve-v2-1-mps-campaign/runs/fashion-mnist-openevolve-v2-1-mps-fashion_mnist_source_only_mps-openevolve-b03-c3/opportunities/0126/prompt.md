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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.22171137481928, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20909543914794923, "validation_score": 9280.41353228522}
prior_hypothesis: AdamW beta2=0.95 will exceed 9,283 correct predictions by continuing the verified monotonic improvement from beta2=0.99 through 0.96 while remaining stable over 522 optimizer steps.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 55.24326145788655, "validation_accuracy": 0.9277, "validation_correct": 9277, "validation_cross_entropy": 0.20867646484375, "validation_score": 9277.413675631604}
prior_hypothesis: Restoring AdamW beta2=0.96 and increasing the cosine schedule’s peak learning rate by 5% will exceed 9,283 correct predictions by accelerating convergence during the fixed 522-step run without materially destabilizing the proven model.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 34.46409162483178, "validation_accuracy": 0.9283, "validation_correct": 9283, "validation_cross_entropy": 0.20936809196472167, "validation_score": 9283.4134390541}
prior_hypothesis: AdamW beta2=0.96 will exceed 9,278 correct predictions by continuing the verified improvement from beta2=0.99 through 0.98 to 0.97 while remaining stable over 522 optimizer steps.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 42.50887762499042, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20954851112365722, "validation_score": 9278.413377384539}
prior_hypothesis: AdamW beta2=0.97 will exceed 9,277 correct predictions by continuing the improvement observed from beta2=0.99 to 0.98 while remaining stable over the fixed 522-step run.

## Recent verification evidence

RECENT RESULT
hypothesis: AdamW beta2=0.96 with a 2.025e-3 peak learning rate will retain or exceed 9,283 correct predictions while lowering cross-entropy below 0.2093681.
change: Restore the best verified beta2 and raise both the optimizer and cosine-schedule base learning rate by 1.25%.
mechanism: Upper-side peak-learning-rate interpolation
evidence_used: Beta2=0.96 at 2.0e-3 achieved the best 9,283 correct; 2.1e-3 lowered cross-entropy to 0.2086765 but lost six predictions, while 1.95e-3 worsened both accuracy and cross-entropy, motivating a narrow upper-side interpolation.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 46.01391216600314, "validation_accuracy": 0.9275, "validation_correct": 9275, "validation_cross_entropy": 0.2093422607421875, "validation_score": 9275.413447885045}

RECENT RESULT
hypothesis: Restoring beta2=0.96 while annealing the learning rate fully to zero will exceed 9,283 correct predictions by preserving the accuracy-optimal peak rate and reducing boundary-crossing updates late in the fixed 522-step run.
change: Restore the best verified AdamW beta2 and replace the 5% cosine learning-rate floor with a zero floor.
mechanism: Zero-floor terminal cosine annealing
evidence_used: Beta2=0.96 at a 2.0e-3 peak achieved the best 9,283 correct; changing the peak in either direction regressed, so isolating the previously untested terminal learning rate is the most direct schedule refinement.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 47.407488375203684, "validation_accuracy": 0.9252, "validation_correct": 9252, "validation_cross_entropy": 0.21438936767578126, "validation_score": 9252.411729559983}

RECENT RESULT
hypothesis: Restoring the accuracy-optimal 2.0e-3 peak and raising the cosine floor from 5% to 10% will exceed 9,283 correct predictions by sustaining useful late-run optimization without disturbing the proven peak rate.
change: Restore AdamW’s peak learning rate to 2.0e-3 and increase the terminal cosine floor to 10%.
mechanism: Elevated terminal cosine learning-rate floor
evidence_used: Beta2=0.96 with a 2.0e-3 peak and 5% floor achieved the best 9,283 correct, while annealing to zero sharply regressed to 9,252 and worsened cross-entropy, motivating an upper-side test of the terminal floor.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 37.17911333311349, "validation_accuracy": 0.927, "validation_correct": 9270, "validation_cross_entropy": 0.20819628524780273, "validation_score": 9270.413840040816}

RECENT RESULT
hypothesis: Matching training translations to the verified vertical-biased, distance-stratified TTA kernel will exceed 9,283 correct predictions by training the model on the same nuisance distribution used for inference averaging.
change: Replace the symmetric training translation probabilities with the exact normalized TTA weights while preserving the best architecture, optimizer, loss, and schedule.
mechanism: Train–test translation-kernel alignment
evidence_used: The best result used vertical-biased, distance-stratified TTA and reached 9,283 correct; nearby beta2, peak-rate, and cosine-floor changes regressed, while training augmentation remains mismatched at symmetric 0.09 axial weights and uniform radius-two weights.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 36.76095829112455, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2084848720550537, "validation_score": 9281.413741215601}

RECENT RESULT
hypothesis: Halfway interpolation toward the inference translation kernel will retain or exceed 9,283 correct predictions while improving cross-entropy relative to the beta2=0.96 baseline.
change: Restore AdamW beta2=0.96 and replace symmetric training translations with a 50% interpolation toward the vertical-biased, distance-stratified TTA weights.
mechanism: Partial train–test translation-kernel alignment
evidence_used: Full kernel alignment reached 9,281 correct but lowered cross-entropy from 0.2093681 to 0.2084849; partial alignment tests whether that confidence gain can be captured without losing the baseline’s two additional correct predictions.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 43.278973957989365, "validation_accuracy": 0.9273, "validation_correct": 9273, "validation_cross_entropy": 0.20902822265625, "validation_score": 9273.413555275742}

RECENT RESULT
hypothesis: Restoring beta2=0.96 and evaluating a decay-0.95 EMA of the final training trajectory will exceed 9,283 correct predictions by suppressing late boundary-crossing parameter noise.
change: Restore the best verified AdamW beta2 and maintain an optimizer-updated parameter EMA that is substituted when the model enters evaluation.
mechanism: Short-horizon exponential weight averaging
evidence_used: Beta2=0.96 achieved the best 9,283 correct, while changing peak rate or terminal floor consistently lost accuracy; this motivates smoothing late updates without altering the proven schedule.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 54.74098879098892, "validation_accuracy": 0.9278, "validation_correct": 9278, "validation_cross_entropy": 0.20993251342773436, "validation_score": 9278.4132461889}

RECENT RESULT
hypothesis: At the verified beta2=0.96 and 2.0e-3 learning rate, explicitly adding translation-tolerant second-order channel statistics while preserving the spatial flattening path will exceed 9,283 correct predictions by improving discrimination between classes with similar outlines but different part and texture co-occurrences.
change: Restore the best learning rate and augment the 4×4 representation with the unique entries of a centered covariance matrix from a learned 12-channel projection; reduce the classifier width from 140 to 133 to remain below the parameter ceiling.
mechanism: Learned low-rank channel-covariance augmentation
evidence_used: The unmodified representation reached 9,283 correct at 2.0e-3, whereas 2.1e-3 fell to 9,277 and non-local attention fell to 9,265. This tests a different assumption: rather than replacing the proven spatial path or learning unstable pairwise attention, it supplies compact explicit quadratic evidence alongside that path.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249189, "training_seconds": 63.60190404183231, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.2123107593536377, "validation_score": 9239.412435504793}

RECENT RESULT
hypothesis: Adding a 5×5 contrast channel will exceed 9,283 correct predictions by exposing medium-scale garment structure that the initial 3×3 convolution cannot directly observe, while preserving the verified backbone and optimization settings.
change: Expand the fixed image basis from five to six channels with 5×5 local contrast, adding only 216 learned parameters.
mechanism: Deterministic multi-scale local-contrast basis
evidence_used: The 245,150-parameter baseline reached 9,283 correct, while learned non-local attention fell to 9,265 and covariance augmentation to 9,239; this motivates a low-capacity deterministic feature that supplies broader context without replacing or compressing proven components.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245366, "training_seconds": 53.25934004201554, "validation_accuracy": 0.9266, "validation_correct": 9266, "validation_cross_entropy": 0.2096618492126465, "validation_score": 9266.413338653547}

RECENT RESULT
hypothesis: Restoring beta2=0.96 and averaging translation predictions in probability space will exceed 9,283 correct predictions by limiting domination from overconfident atypical shifts while retaining the verified TTA kernel.
change: Restore the best verified AdamW beta2 and replace logit averaging across translations with weighted softmax-probability averaging; preserve logit averaging across horizontal flips.
mechanism: Probability-space translation marginalization
evidence_used: Beta2=0.96 produced the best 9,283-correct result, while nearby optimizer and augmentation changes regressed; full translation-kernel alignment lowered cross-entropy, motivating refinement of how the established translation views are combined.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 37.896887791808695, "validation_accuracy": 0.9271, "validation_correct": 9271, "validation_cross_entropy": 0.20914803009033203, "validation_score": 9271.413514298958}

RECENT RESULT
hypothesis: AdamW beta2=0.96 with batch size 128 will exceed 9,283 correct predictions by providing roughly 50% more optimizer updates and noisier gradients over the same 100,000 examples.
change: Restore the best verified beta2 and reduce batch size from 192 to 128 while preserving the architecture, learning-rate schedule, augmentation, loss, and TTA.
mechanism: Higher-update stochastic optimization at the verified beta2 optimum
evidence_used: Beta2=0.96 at batch size 192 achieved the best result of 9,283 correct; nearby beta2, peak-rate, cosine-floor, representation, and TTA changes regressed, while batch size and its resulting update count remain untested.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Batch size 160 at the verified beta2=0.96 and 2.0e-3 peak learning rate will exceed 9,283 correct predictions by providing 20% more optimizer updates while avoiding the runtime failure observed at batch size 128.
change: Reduce batch size from 192 to 160 and restore the accuracy-optimal 2.0e-3 optimizer and cosine-schedule learning rate.
mechanism: Intermediate higher-update stochastic optimization
evidence_used: Batch size 192 with beta2=0.96 and 2.0e-3 achieved the best 9,283 correct; batch size 128 was the only higher-update test but timed out, so 160 isolates a moderate 522-to-625-step increase with substantially less runtime risk.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 160, "examples_processed": 100000, "optimizer_steps": 626, "parameters": 245150, "training_seconds": 59.59176583401859, "validation_accuracy": 0.9256, "validation_correct": 9256, "validation_cross_entropy": 0.20767303161621095, "validation_score": 9256.41401934705}

RECENT RESULT
hypothesis: Adding a modest symmetric KL penalty between original and flipped predictions late in training will exceed 9,283 correct predictions by making the two views agree before their logits are averaged at validation.
change: Preserve the verified architecture, optimizer, augmentation, schedule, and TTA while adding a symmetric flip-consistency loss that ramps from zero to 0.10.
mechanism: Late-ramped horizontal-flip consistency regularization
evidence_used: The 9,283-correct baseline already trains and evaluates a horizontal-flip logit ensemble, while optimizer, schedule, translation, representation, and probability-averaging changes regressed; explicit agreement between the paired predictions remains untested.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245150, "training_seconds": 41.10045158304274, "validation_accuracy": 0.928, "validation_correct": 9280, "validation_cross_entropy": 0.20919101409912108, "validation_score": 9280.41349959946}



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
