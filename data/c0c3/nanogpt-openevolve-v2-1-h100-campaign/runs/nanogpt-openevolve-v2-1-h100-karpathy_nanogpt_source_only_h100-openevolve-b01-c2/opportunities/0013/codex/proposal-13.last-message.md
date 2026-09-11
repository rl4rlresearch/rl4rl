MECHANISM: Four-step Muon polar orthogonalization

HYPOTHESIS: Reducing Muon orthogonalization from five iterations to four will preserve update quality while cutting optimizer overhead, process more than 497M tokens, and lower val_bpb below 0.995558.

INTENDED_EDIT: Keep the proven SSSL model, 524K-token batch, and learning-rate schedule unchanged; remove one Muon polar iteration per optimizer step.

EVIDENCE: The 524K-token design achieved 0.995558 on 497.0M tokens, outperforming the one-microbatch design’s 0.996687 on 408.9M tokens despite fewer updates, indicating that additional throughput is more valuable than update frequency. Muon iteration count is an untested throughput lever that does not reduce model capacity.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=4, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE