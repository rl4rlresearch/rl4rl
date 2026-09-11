# Improve fixed-time language-model pretraining

You are an autonomous ML engineer improving the source code for single-GPU
language-model pretraining.

## Goal

Minimize validation bits per byte (`val_bpb`) after a fixed five-minute training
window on the supplied H100 worker. Lower is better. Startup, compilation, and
final validation are outside the measured training window, and every submitted
version starts from a fresh initialization.

You may change the architecture, optimizer, schedules, batching, numerical
implementation, or other contents of `train.py`. The fixed data preparation,
tokenizer, validation procedure, hardware class, and time accounting are not
editable. A useful change must produce a complete trainable implementation and
finish with the required summary metrics.

## Work boundaries

Minimize val_bpb. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: val_bpb, training_seconds, peak_vram_mb, mfu_percent, total_tokens_M, num_steps, num_params_M, depth.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or validation yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"depth": 8.0, "mfu_percent": 36.66, "num_params_M": 50.3, "num_steps": 1951.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.4, "training_seconds": 300.0, "val_bpb": 0.983689}
prior_hypothesis: Reallocating the redundant late 512-token window to layer 2 will lower val_bpb below 0.983718 by giving the first global-consolidation stage two consecutive wider-context layers whose features propagate through the remaining network.

## Recent verification evidence

RECENT RESULT
hypothesis: Using 512-token attention only in the local layers immediately preceding full-context layers 4 and 8 will lower val_bpb below 0.984467 by enriching inputs to global consolidation while retaining more throughput than uniformly widening all six local layers.
change: Add a medium 512-token window type and change the pattern from SSSL to SSML, yielding 256, 256, 512, 2048-token windows in each four-layer stage.
mechanism: Pre-consolidation context widening
evidence_used: Uniform 512-token local windows nearly matched the best result at 0.984862 but reduced total tokens to 503.6M, while 256-token windows with full-context layers 4 and 8 achieved 0.984467 at 516.9M tokens; concentrating wider attention immediately before those successful global layers tests whether its contextual benefit can be retained at lower cost.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.65, "num_params_M": 50.3, "num_steps": 1951.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.4, "training_seconds": 300.1, "val_bpb": 0.98372}

RECENT RESULT
hypothesis: Retaining the 512-token window before layer 4 while reverting layer 7 to 256 tokens will lower val_bpb below 0.983720 because early enriched features can propagate through the second stage, while the late wider window is partly redundant with layer 8’s full-context attention.
change: Change the eight-layer window schedule from SSMLSSML to SSMLSSSL, preserving the first medium window and recovering throughput from the second.
mechanism: Asymmetric pre-consolidation widening
evidence_used: SSML improved val_bpb from 0.984467 to 0.983720 while processing fewer tokens; isolating its earlier widening tests whether that benefit can be retained with throughput closer to the 256-token baseline.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.5, "num_params_M": 50.3, "num_steps": 1957.0, "peak_vram_mb": 44908.2, "total_tokens_M": 513.0, "training_seconds": 300.1, "val_bpb": 0.983718}

RECENT RESULT
hypothesis: Reallocating the redundant late 512-token window to layer 2 will lower val_bpb below 0.983718 by giving the first global-consolidation stage two consecutive wider-context layers whose features propagate through the remaining network.
change: Change the window schedule from SSMLSSSL to SMMLSSSL, widening layers 2 and 3 to 512 tokens while retaining full-context attention at layers 4 and 8.
mechanism: Front-loaded context widening
evidence_used: SSMLSSML achieved 0.983720, while removing its late medium window produced a marginally better 0.983718 with greater throughput, indicating that late widening is redundant and motivating reuse of that compute in the beneficial first stage.
result: improved the objective and became an available design
reported_values: {"depth": 8.0, "mfu_percent": 36.66, "num_params_M": 50.3, "num_steps": 1951.0, "peak_vram_mb": 44908.2, "total_tokens_M": 511.4, "training_seconds": 300.0, "val_bpb": 0.983689}

RECENT RESULT
hypothesis: Widening layer 1 to 512 tokens will lower val_bpb below 0.983689 by letting all three layers before the first full-context layer build wider-context features.
change: Change the attention schedule from SMMLSSSL to MMMLSSSL, retaining full-context layers 4 and 8 while widening the remaining first-stage local layer.
mechanism: Fully widened first consolidation stage
evidence_used: Reallocating the late 512-token window to layer 2 improved val_bpb from 0.983718 to 0.983689, while late widening was redundant; extending the same front-loaded trend to layer 1 is the most direct test of whether first-stage context remains limiting.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.33, "num_params_M": 50.3, "num_steps": 1920.0, "peak_vram_mb": 44908.2, "total_tokens_M": 503.3, "training_seconds": 300.1, "val_bpb": 0.984942}

RECENT RESULT
hypothesis: Reallocating layer 2’s marginal 256-token window increase to layer 3 will lower val_bpb below 0.983689 by providing 768-token context immediately before the first full-context consolidation layer without changing attention compute or throughput.
change: Change the first-stage windows from 256/512/512/2048 to 256/256/768/2048 while retaining 256-token local windows and full-context layers elsewhere.
mechanism: Pre-consolidation context-budget concentration
evidence_used: Widening layer 3 alone improved val_bpb from 0.984467 to 0.983718, whereas also widening layer 2 improved it by only 0.000029; concentrating the same added attention budget in the demonstrably valuable pre-consolidation layer directly tests whether proximity to global mixing matters more than distributed widening.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.42, "num_params_M": 50.3, "num_steps": 1939.0, "peak_vram_mb": 44908.2, "total_tokens_M": 508.3, "training_seconds": 300.1, "val_bpb": 0.984337}

RECENT RESULT
hypothesis: Replacing the additive ReLU² MLP with a parameter-matched multiplicative SwiGLU will lower val_bpb below 0.983689 by allowing each token to conditionally gate predictive features while preserving nearly the same parameter and compute budget.
change: Factor each MLP expansion into learned gate and value branches, with a 64-aligned hidden width chosen to approximately match the existing MLP parameter count.
mechanism: Parameter-matched SwiGLU feature routing
evidence_used: Further static context widening plateaued or regressed—MMMLSSSL reached 0.984942 and concentrating 768 tokens at layer 3 reached 0.984337—suggesting the load-bearing assumption that attention span is the remaining bottleneck should be challenged in favor of more expressive tokenwise feature computation.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 34.9, "num_params_M": 50.1, "num_steps": 1871.0, "peak_vram_mb": 44652.5, "total_tokens_M": 490.5, "training_seconds": 300.0, "val_bpb": 0.986926}

RECENT RESULT
hypothesis: Replacing each value embedding’s per-head scalar gate with a per-feature gate will reduce val_bpb below 0.983689 by enabling finer token-dependent routing with negligible parameter and compute growth.
change: Expand the existing zero-initialized value-residual gate from one coefficient per KV head to one coefficient per KV feature, retaining the SMMLSSSL attention schedule and neutral initialization.
mechanism: Feature-wise value-residual gating
evidence_used: Attention-window reallocations have plateaued at 0.983689, while parameter-matched SwiGLU regressed to 0.986926 alongside substantially lower throughput; this motivates adding expressivity specifically to the existing value-residual pathway at much lower cost.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 35.63, "num_params_M": 50.4, "num_steps": 1893.0, "peak_vram_mb": 45924.8, "total_tokens_M": 496.2, "training_seconds": 300.1, "val_bpb": 0.985862}

RECENT RESULT
hypothesis: Using a 384-token window in layer 2 while retaining layer 3’s 512-token window will lower val_bpb below 0.983689 by preserving useful early context while recovering throughput from layer 2’s marginal widening.
change: Add a 384-token transition window and change the schedule from SMMLSSSL to STMLSSSL.
mechanism: Partial front-loaded context widening
evidence_used: Widening layer 3 alone reached 0.983718 at 513.0M tokens, while also widening layer 2 improved only to 0.983689 and reduced training to 511.4M tokens; interpolating layer 2’s window tests whether its small contextual benefit can be retained with less throughput loss.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.39, "num_params_M": 50.3, "num_steps": 1944.0, "peak_vram_mb": 44908.2, "total_tokens_M": 509.6, "training_seconds": 300.1, "val_bpb": 0.984247}

RECENT RESULT
hypothesis: Replacing linear warmdown with cosine warmdown will lower val_bpb below 0.983689 by sustaining larger productive updates through the middle of training while tapering updates more gently near convergence.
change: Keep the best SMMLSSSL architecture and all optimization magnitudes fixed, changing only the warmdown curve from linear to cosine.
mechanism: Cosine learning-rate warmdown
evidence_used: SMMLSSSL achieved the best val_bpb of 0.983689, while nearby attention reallocations and more expressive MLP/gating changes regressed; preserving that architecture and testing a throughput-neutral optimization change is the most informative next axis.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.56, "num_params_M": 50.3, "num_steps": 1946.0, "peak_vram_mb": 44908.2, "total_tokens_M": 510.1, "training_seconds": 300.1, "val_bpb": 0.986693}

RECENT RESULT
hypothesis: Extending linear warmdown from 62.5% to 75% of training will lower val_bpb below 0.983689 by reducing mid-training update magnitudes while preserving the proven linear decay shape.
change: Keep the best SMMLSSSL architecture and all optimizer magnitudes fixed, but begin linear warmdown at 25% rather than 37.5% of the time budget.
mechanism: Earlier linear learning-rate decay
evidence_used: Cosine warmdown regressed from 0.983689 to 0.986693 while sustaining higher learning rates during early warmdown; an earlier linear decay directly tests whether smaller mid-training updates improve convergence without changing throughput or architecture.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Starting linear warmdown at 25% of training will lower val_bpb below 0.983689 by reducing mid-training update magnitudes while retaining the proven linear decay shape.
change: Increase WARMDOWN_RATIO from 0.625 to 0.75, leaving the best SMMLSSSL architecture and all optimizer magnitudes unchanged.
mechanism: Earlier linear learning-rate decay
evidence_used: Cosine warmdown regressed to 0.986693 while sustaining larger mid-training learning rates; the previously proposed earlier linear decay was not verified, so implementing it cleanly is the most direct follow-up.
result: was valid but was not a strict improvement
reported_values: {"depth": 8.0, "mfu_percent": 36.07, "num_params_M": 50.3, "num_steps": 1920.0, "peak_vram_mb": 44908.2, "total_tokens_M": 503.3, "training_seconds": 300.1, "val_bpb": 0.984253}



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
