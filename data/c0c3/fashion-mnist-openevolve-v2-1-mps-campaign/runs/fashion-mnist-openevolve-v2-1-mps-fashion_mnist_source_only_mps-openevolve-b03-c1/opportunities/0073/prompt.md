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
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 78.55299241608009, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.2404482681274414, "validation_score": 9162.403080090357}
prior_hypothesis: Applying the learned channel gate to both global-average and global-maximum descriptors will exceed 9,153 correct predictions by capturing distributed texture and localized salient features without changing initialization or parameter count.

## Recent verification evidence

RECENT RESULT
hypothesis: Applying the learned channel gate to both global-average and global-maximum descriptors will exceed 9,153 correct predictions by capturing distributed texture and localized salient features without changing initialization or parameter count.
change: Average squeeze-and-excitation signals computed from mean-pooled and max-pooled final features using the existing shared gate.
mechanism: Shared dual-statistic channel attention
evidence_used: Mean-only channel attention improved validation correctness from 9,133 to 9,153; enriching that successful mechanism with complementary extrema information is the most direct next test, while avoiding the runtime-heavy mixed pooling attempted throughout the backbone.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 78.55299241608009, "validation_accuracy": 0.9162, "validation_correct": 9162, "validation_cross_entropy": 0.2404482681274414, "validation_score": 9162.403080090357}

RECENT RESULT
hypothesis: Explicitly gating the max-minus-mean channel concentration descriptor will exceed 9,162 correct predictions by distinguishing broad texture responses from isolated salient activations while preserving identity initialization and the parameter count.
change: Add a parameter-free spatial concentration descriptor to the successful shared channel gate and equally combine its signal with the existing average- and maximum-pooled signals.
mechanism: Shared mean–maximum–concentration channel attention
evidence_used: Mean-only attention reached 9,153 correct and adding maximum context raised this to 9,162, showing complementary spatial statistics help; max-minus-mean exposes their otherwise implicit interaction without the runtime-heavy backbone pooling that timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Allowing each feature channel to learn its own mean-versus-maximum balance will exceed 9,162 correct predictions while preserving the successful equal fusion at initialization.
change: Add 96 trainable fusion logits and replace the fixed 50/50 channel-signal average with a sigmoid-bounded, per-channel mixture; total parameters become 249,858.
mechanism: Learned per-channel fusion of mean and maximum attention
evidence_used: Mean-only attention achieved 9,153 correct, while equal mean–maximum fusion improved to 9,162, showing both descriptors are useful; learning their balance per channel directly refines that successful mechanism without the extra gate evaluation that caused the concentration variant to time out.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249858, "training_seconds": 75.04472808307037, "validation_accuracy": 0.9141, "validation_correct": 9141, "validation_cross_entropy": 0.24070005950927734, "validation_score": 9141.402998288078}

RECENT RESULT
hypothesis: Replacing the correlated maximum descriptor with max-minus-mean concentration will exceed 9,162 correct predictions by separating distributed activation strength from localized salience while preserving parameter count, identity initialization, and two gate evaluations.
change: Feed the shared channel gate average-pooled features and the parameter-free max-minus-mean concentration descriptor, then retain equal signal fusion.
mechanism: Disentangled mean–concentration channel attention
evidence_used: Mean-only attention reached 9,153 correct and mean–maximum attention reached 9,162, showing spatial extrema add useful information; the three-descriptor concentration variant timed out, so substituting concentration for the redundant raw maximum tests that signal without its extra gate evaluation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A single globally learned fusion weight will exceed 9,162 correct predictions by adapting the successful mean–maximum balance without the overfitting and instability observed with 96 independent channel weights.
change: Add one zero-initialized scalar fusion logit and replace the fixed 50/50 gate average with its sigmoid-bounded learned mixture, increasing parameters to 249,763 while preserving the current computation exactly at initialization.
mechanism: Learned global mean–maximum attention fusion
evidence_used: Fixed equal fusion improved mean-only attention from 9,153 to 9,162 correct, but per-channel learned fusion fell to 9,141; this motivates testing the same adaptation with one strongly shared degree of freedom.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 40/60 mean-to-maximum gate mixture will exceed 9,162 correct predictions by modestly extending the observed gain from adding localized maximum context, without the instability of learned fusion weights.
change: Replace equal channel-signal fusion with a parameter-free mixture favoring maximum-pooled context while preserving identity initialization and parameter count.
mechanism: Fixed max-biased dual-statistic channel attention
evidence_used: Mean-only attention achieved 9,153 correct and equal mean–maximum fusion improved this to 9,162, whereas learned per-channel fusion fell to 9,141 and learned global fusion timed out; a fixed max-biased mixture is the cleanest next probe.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 70.66292250016704, "validation_accuracy": 0.9155, "validation_correct": 9155, "validation_cross_entropy": 0.2416100830078125, "validation_score": 9155.402702915226}

RECENT RESULT
hypothesis: A 60/40 mean-to-maximum gate mixture will exceed 9,162 correct predictions by retaining complementary maximum context while emphasizing the stronger average-pooled descriptor.
change: Replace equal channel-signal fusion with a parameter-free mixture favoring average-pooled context, preserving initialization, parameter count, and runtime.
mechanism: Fixed mean-biased dual-statistic channel attention
evidence_used: Equal mean–maximum fusion achieved 9,162 correct, while the symmetric 40/60 max-biased mixture fell to 9,155 and mean-only attention achieved 9,153; testing the unexplored mean-biased direction is the most informative refinement of the best mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 78.24348479113542, "validation_accuracy": 0.9152, "validation_correct": 9152, "validation_cross_entropy": 0.24071038360595703, "validation_score": 9152.402994934682}

RECENT RESULT
hypothesis: A 45/55 mean-to-maximum gate mixture will exceed 9,162 correct predictions by preserving the successful near-equal fusion while modestly favoring the empirically less harmful maximum-biased direction.
change: Replace equal channel-signal fusion with a parameter-free 45/55 mixture favoring maximum-pooled context.
mechanism: Local max-biased dual-statistic channel attention
evidence_used: Equal fusion achieved 9,162 correct; the 40/60 max-biased variant retained 9,155 versus 9,152 for the symmetric 60/40 mean-biased variant, motivating a midpoint probe between the best mixture and the stronger neighboring direction.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249762, "training_seconds": 77.302411041921, "validation_accuracy": 0.9151, "validation_correct": 9151, "validation_cross_entropy": 0.24037016906738282, "validation_score": 9151.403105470019}

RECENT RESULT
hypothesis: Sharpening the gate response by 25% will exceed 9,162 correct predictions by helping the zero-initialized attention learn stronger channel distinctions within 1,564 optimizer steps while retaining its bounded scale range.
change: Multiply the fused mean–maximum gate signal by 1.25 before `tanh`, preserving identity initialization, parameter count, descriptors, and runtime.
mechanism: Temperature-sharpened dual-statistic channel attention
evidence_used: Dual-statistic attention improved correctness from 9,153 to 9,162, while changing its equal descriptor mixture reduced accuracy; strengthening the successful gate response is therefore a cleaner next test than altering descriptor fusion again.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Replacing global pooled channel scaling with a depthwise spatial residual block will exceed 9,162 correct predictions by learning new relationships among neighboring coarse image regions rather than repeatedly refining fixed mean/maximum summaries.
change: Replace squeeze-and-excitation with an identity-initialized bottleneck residual block that mixes channels and the full 3×3 feature layout, using 249,845 learned parameters.
mechanism: Spatially mixing residual feature refinement
evidence_used: Dual-statistic attention improved 9,133 to 9,162 correct, proving an extra learned feature-transformation branch is useful, but every subsequent descriptor-fusion refinement failed or timed out; this challenges the load-bearing assumption that further progress should come from global channel reweighting.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Jointly encoding mean and maximum descriptors through one regularized bottleneck will exceed 9,162 correct predictions by learning cross-statistic channel relationships while avoiding unstable explicit fusion weights.
change: Concatenate mean and maximum contexts and process them with a single 192→16→96 identity-initialized channel gate, reducing gate computation and parameters to 249,754.
mechanism: Joint low-rank dual-statistic channel attention
evidence_used: Equal mean–maximum attention improved correctness from 9,153 to 9,162, but fixed reweighting and 96 independent fusion weights regressed; a joint low-rank encoding preserves both successful descriptors while learning their interaction with stronger parameter sharing.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 249754, "training_seconds": 72.145049582934, "validation_accuracy": 0.9134, "validation_correct": 9134, "validation_cross_entropy": 0.2436003372192383, "validation_score": 9134.4020584307}

RECENT RESULT
hypothesis: Adding a lightweight spatial gate after the successful dual-statistic channel gate will exceed 9,162 correct predictions by emphasizing class-relevant regions within the final 3×3 feature map while preserving the current network exactly at initialization.
change: Add a zero-initialized 3×3 spatial gate driven by channel-wise mean and maximum feature maps, increasing the model by 19 parameters to 249,781.
mechanism: Identity-initialized spatial saliency attention
evidence_used: Dual-statistic channel attention improved correctness from 9,153 to 9,162, but further global descriptor and fusion changes regressed or timed out; spatial attention tests a complementary axis with far less computation than the timed-out residual spatial block.
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
