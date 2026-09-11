MECHANISM: Quadratically refined output-gate learning rate

HYPOTHESIS: AdamW at 0.0115 for only the four output-gate matrices will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

INTENDED_EDIT: Raise the dedicated output-gate AdamW learning rate from 0.0100 to 0.0115 without changing the proven 704/1024/1344 backbone or other optimizer groups.

EVIDENCE: Output-gate AdamW achieved 0.991514 at 0.0100, versus 0.992015 at 0.0075 and 0.991889 at 0.0150; quadratic interpolation of these bracketing results places the estimated minimum near 0.0115.

<<<<<<< SEARCH
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.023, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE