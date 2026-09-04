MECHANISM: Best-verified adaptive output-gate optimization

HYPOTHESIS: Restoring the output-gate AdamW learning rate to 0.0115 will lower val_bpb below 0.991889 while retaining at least 470M trained tokens.

INTENDED_EDIT: Reduce the dedicated learning rate for the four output-gate matrices from 0.015 to the best-verified 0.0115, preserving the 704/1024/1344 backbone and all other optimizer groups.

EVIDENCE: Output-gate AdamW at 0.0115 achieved the best observed val_bpb of 0.991177 over 480.2M tokens, outperforming both the current 0.015 setting at 0.991889 and 0.010 at 0.991514.

<<<<<<< SEARCH
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.03, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.023, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE