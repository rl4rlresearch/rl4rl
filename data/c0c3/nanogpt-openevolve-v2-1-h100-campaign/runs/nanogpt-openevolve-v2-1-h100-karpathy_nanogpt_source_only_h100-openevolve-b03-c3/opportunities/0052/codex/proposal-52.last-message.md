MECHANISM: Context-scale-specific output-gate optimization

HYPOTHESIS: Using AdamW at 0.0115 for full-context output gates and 0.0100 for pre-handoff short-context gates will lower val_bpb below 0.991177 while retaining at least 470M training tokens.

INTENDED_EDIT: Split the four output-gate matrices into full-context and pre-handoff optimizer groups, preserving the best verified rate for global branches while applying the proven conservative rate to short-context branches.

EVIDENCE: Joint output-gate AdamW improved val_bpb to 0.991514 at 0.0100 and the best 0.991177 at 0.0115, while 0.0150 regressed to 0.991889; separating gates by their different attention scales tests whether the shared optimum masks role-specific sensitivity without changing compute.

<<<<<<< SEARCH
        output_gate_params = [
            block.attn.out_gate.weight for block in self.transformer.h
            if block.attn.out_gate is not None
        ]
        output_gate_ids = {id(p) for p in output_gate_params}
=======
        full_context_output_gate_params = [
            block.attn.out_gate.weight
            for i, block in enumerate(self.transformer.h)
            if block.attn.out_gate is not None
            and self.window_sizes[i][0] == self.config.sequence_len
        ]
        pre_handoff_output_gate_params = [
            block.attn.out_gate.weight
            for i, block in enumerate(self.transformer.h)
            if block.attn.out_gate is not None
            and self.window_sizes[i][0] != self.config.sequence_len
        ]
        output_gate_params = (
            full_context_output_gate_params + pre_handoff_output_gate_params
        )
        output_gate_ids = {id(p) for p in output_gate_params}
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.0225, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=full_context_output_gate_params, lr=scalar_lr * 0.023, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=pre_handoff_output_gate_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE