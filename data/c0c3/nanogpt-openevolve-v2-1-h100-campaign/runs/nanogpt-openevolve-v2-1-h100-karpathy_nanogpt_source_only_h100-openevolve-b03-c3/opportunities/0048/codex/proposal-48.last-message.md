MECHANISM: Lower-flank output-gate learning-rate refinement

HYPOTHESIS: AdamW at 0.01125 for only the four output-gate matrices will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Raise the dedicated output-gate AdamW learning rate from 0.0075 to the untested 0.01125 midpoint between 0.0100 and the best-verified 0.0115, preserving the 704/1024/1344 backbone and all other optimizer groups.

EVIDENCE: Output-gate AdamW at 0.0115 produced the two best results, 0.991177 and 0.991333, while 0.0100 reached 0.991514 and 0.0120 regressed; testing 0.01125 probes the narrow lower side of the apparent optimum without repeating the unsuccessful higher rate.

<<<<<<< SEARCH
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.015, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.0225, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE