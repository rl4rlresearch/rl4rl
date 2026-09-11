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
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.00435666600242, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.23423449478149413, "validation_score": 9240.405109403533}
prior_hypothesis: Adding a cosine label-smoothing decay from 0.05 to 0.01 to Reference Design 3’s 55%–95% flip-ensemble curriculum will exceed 9,237 correct predictions by regularizing representation learning early while sharpening class margins late, without changing average smoothing.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 30.37924558413215, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2238097038269043, "validation_score": 9240.408560251186}
prior_hypothesis: Increasing evaluation-logit scaling from 1.05 to 1.10 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.228181.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 30.817835124908015, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2420552200317383, "validation_score": 9237.402558591548}
prior_hypothesis: Annealing flip-ensemble weight from 60% to 90% will exceed 9,235 correct predictions by preserving the successful 75% average curriculum while avoiding the weaker 50% and 100% endpoint regimes.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 27.830069333082065, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.24180156936645508, "validation_score": 9237.402640818254}
prior_hypothesis: A cosine curriculum from 55% to 95% flip-ensemble weight will exceed 9,237 correct predictions by retaining useful endpoint diversity while avoiding the weaker extremes of the 50%–100% schedule.

## Recent verification evidence

RECENT RESULT
hypothesis: Annealing flip-ensemble weight from 60% to 90% will exceed 9,235 correct predictions by preserving the successful 75% average curriculum while avoiding the weaker 50% and 100% endpoint regimes.
change: Replace pure flip-ensemble loss with a cosine curriculum centered at 75%, mixing individual-view and flip-averaged cross-entropy throughout training.
mechanism: Bounded progressive flip-orbit supervision
evidence_used: The 50%-to-100% curriculum achieved the best result at 9,235 correct, while static 75% reached 9,233 and static 100% fell to 9,230; narrowing the curriculum around the empirically favored weight tests whether progression helps without spending training at weaker extremes.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 30.817835124908015, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.2420552200317383, "validation_score": 9237.402558591548}

RECENT RESULT
hypothesis: A cosine curriculum from 65% to 85% flip-ensemble weight will exceed 9,237 correct predictions by preserving beneficial progression while concentrating training closer to the empirically favored 75% regime.
change: Replace the static 80% ensemble-loss mixture with a cosine schedule centered at 75% and bounded between 65% and 85%.
mechanism: Narrow progressive flip-orbit supervision
evidence_used: Narrowing the curriculum from 50%–100% (9,235 correct) to 60%–90% (9,237 correct) improved accuracy, while static 75% reached 9,233; this motivates testing a narrower schedule without eliminating progression.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 37.1608796659857, "validation_accuracy": 0.9229, "validation_correct": 9229, "validation_cross_entropy": 0.24190288772583007, "validation_score": 9229.402607969545}

RECENT RESULT
hypothesis: Replacing the flattened single-scale classifier with dilated multiscale residual features, class-specific spatial attention, and coarse average/max shape pooling will exceed 9,237 correct predictions by learning translation-tolerant local evidence while retaining coarse spatial structure.
change: Restore the strongest weighted-translation, flip-complete training recipe, but replace its load-bearing sequential 7×7 flattening assumption with a 242,910-parameter multiscale encoder and dual local/global prediction head.
mechanism: Class-conditional multiscale evidence pooling
evidence_used: Differential-basis encoding raised the flip-complete design from 9,186 to 9,224 correct, but subsequent loss-schedule tuning plateaued at 9,237 and another conventional residual refinement reached only 9,039; this motivates reallocating capacity from absolute-position flattening to a genuinely different class-conditional evidence computation.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Replacing multinomial translation draws with randomly permuted systematic samples will exceed 9,237 correct predictions by preserving the proven translation distribution while reducing per-batch augmentation and BatchNorm noise.
change: Sample nearly exact reliability-weighted translation proportions in every minibatch, while randomly assigning offsets to examples.
mechanism: Stratified translation-orbit sampling
evidence_used: Reliability-weighted translation sampling contributed to the 9,186-correct flip-complete design and remains in the 9,237-correct best design; with only 522 updates, reducing multinomial count variance is a targeted refinement of that proven invariance recipe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 51.65910162497312, "validation_accuracy": 0.923, "validation_correct": 9230, "validation_cross_entropy": 0.24575444412231445, "validation_score": 9230.40136320794}

RECENT RESULT
hypothesis: A cosine curriculum from 55% to 95% flip-ensemble weight will exceed 9,237 correct predictions by retaining useful endpoint diversity while avoiding the weaker extremes of the 50%–100% schedule.
change: Narrow the current ensemble-loss curriculum from 50%–100% to 55%–95%, preserving its 75% average and progression direction.
mechanism: Intermediate-width progressive flip-orbit supervision
evidence_used: The 60%–90% curriculum achieved 9,237 correct versus 9,235 for 50%–100%, while narrowing further to 65%–85% fell to 9,229; testing the unmeasured intermediate width directly brackets the strongest region.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 27.830069333082065, "validation_accuracy": 0.9237, "validation_correct": 9237, "validation_cross_entropy": 0.24180156936645508, "validation_score": 9237.402640818254}

RECENT RESULT
hypothesis: A cosine curriculum from 57.5% to 92.5% flip-ensemble weight will exceed 9,237 correct predictions by interpolating between the two strongest schedules while avoiding both the inferior narrow and widest regimes.
change: Widen the current 65%–85% curriculum to 57.5%–92.5%, preserving its 75% center and progression direction.
mechanism: Mid-width progressive flip-orbit supervision
evidence_used: The 60%–90% and 55%–95% schedules both reached 9,237 correct, with 55%–95% achieving lower cross-entropy; 65%–85% fell to 9,229 and 50%–100% reached 9,235, motivating a midpoint within the strongest observed width interval.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 31.865320916986093, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.24207230072021485, "validation_score": 9235.402553055656}

RECENT RESULT
hypothesis: Adding a cosine label-smoothing decay from 0.05 to 0.01 to Reference Design 3’s 55%–95% flip-ensemble curriculum will exceed 9,237 correct predictions by regularizing representation learning early while sharpening class margins late, without changing average smoothing.
change: Restore the strongest differential-basis, weighted-translation, flip-complete design and replace its static 0.03 label smoothing with a cosine decay centered at 0.03.
mechanism: Coupled target-regularization curriculum
evidence_used: Reference Design 3 tied the best correct count at 9,237 and achieved the lowest cross-entropy; progressive flip supervision also improved over static 75% supervision, motivating a complementary curriculum for target smoothing.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 44.00435666600242, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.23423449478149413, "validation_score": 9240.405109403533}

RECENT RESULT
hypothesis: Widening label smoothing from 0.06 early to 0.00 late, while preserving its 0.03 average, will exceed 9,240 correct predictions by strengthening early regularization and optimizing hard-label margins near convergence.
change: Increase the cosine label-smoothing amplitude from 0.02 to 0.03, changing the schedule from 0.05→0.01 to 0.06→0.00.
mechanism: Wider cosine target-sharpening curriculum
evidence_used: The current 0.05→0.01 curriculum improved Reference Design 3 from 9,237 to 9,240 correct and reduced cross-entropy from 0.24180 to 0.23423; widening the same successful curriculum directly tests whether stronger early smoothing and complete late sharpening extend that gain.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 34.96090649999678, "validation_accuracy": 0.9232, "validation_correct": 9232, "validation_cross_entropy": 0.2313438129425049, "validation_score": 9232.406060431493}

RECENT RESULT
hypothesis: Increasing initial label smoothing from 0.05 to 0.06 while retaining the successful 0.01 terminal floor will exceed 9,240 correct predictions by strengthening early regularization without the harmful complete sharpening of the inferior 0.06→0.00 schedule.
change: Replace static 0.03 smoothing with a cosine decay from 0.06 to 0.01 for both individual-view and flip-ensemble losses.
mechanism: Floor-preserving target-smoothing curriculum
evidence_used: The 0.05→0.01 schedule improved accuracy from 9,237 to 9,240, whereas widening both endpoints to 0.06→0.00 fell to 9,232; changing only the initial endpoint isolates whether stronger early regularization is beneficial when late smoothing remains nonzero.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 29.270436957944185, "validation_accuracy": 0.9233, "validation_correct": 9233, "validation_cross_entropy": 0.23572471733093261, "validation_score": 9233.40462086174}

RECENT RESULT
hypothesis: A cosine label-smoothing schedule from 0.045 to 0.015 with the proven 55%–95% flip-ensemble curriculum will exceed 9,240 correct predictions by retaining progressive regularization while avoiding the harmful extremes of the 0.06→0.00 and 0.06→0.01 schedules.
change: Restore the strongest 55%–95% ensemble-loss curriculum and replace static 0.03 smoothing with a narrower cosine decay centered at 0.03.
mechanism: Intermediate-width target-smoothing curriculum
evidence_used: Static 0.03 smoothing with progressive flip supervision reached 9,237 correct, 0.05→0.01 improved this to 9,240, and the wider 0.06→0.00 schedule fell to 9,232; testing amplitude 0.015 brackets the successful region between static and amplitude 0.02 smoothing while avoiding the failed wider regime.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 42.225944832898676, "validation_accuracy": 0.9235, "validation_correct": 9235, "validation_cross_entropy": 0.23561809120178223, "validation_score": 9235.404655777995}

RECENT RESULT
hypothesis: Restoring the 9,240-correct smoothing and flip curricula while scaling evaluation logits by 1.05 will retain 9,240 correct predictions and reduce cross-entropy below 0.234234, strictly improving validation_score.
change: Restore Reference Design 1’s 0.05→0.01 label-smoothing and 55%→95% ensemble-weight schedules, then apply a modest confidence correction to the averaged validation logits.
mechanism: Flip-orbit curriculum with confidence-calibrated translation ensembling
evidence_used: Reference Design 1 achieved the best accuracy at 9,240 correct; the 0.06→0.00 result produced the lowest cross-entropy (0.231344), indicating that sharper confidence can improve the tie-breaker, while constant positive logit scaling preserves every argmax.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 33.20328233297914, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2281807456970215, "validation_score": 9240.407106203018}

RECENT RESULT
hypothesis: Increasing evaluation-logit scaling from 1.05 to 1.10 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.228181.
change: Increase the positive evaluation-only logit multiplier to 1.10 without changing training or class decisions.
mechanism: Stronger post-ensemble confidence calibration
evidence_used: Scaling logits by 1.05 preserved 9,240 correct predictions and reduced cross-entropy from 0.234234 to 0.228181; this motivates a second equal calibration step in the same beneficial direction.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 242546, "training_seconds": 30.37924558413215, "validation_accuracy": 0.924, "validation_correct": 9240, "validation_cross_entropy": 0.2238097038269043, "validation_score": 9240.408560251186}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
