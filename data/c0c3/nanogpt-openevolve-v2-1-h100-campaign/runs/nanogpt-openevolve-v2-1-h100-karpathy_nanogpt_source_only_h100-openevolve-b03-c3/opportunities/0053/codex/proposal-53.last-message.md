MECHANISM: Context-scale-specific output-gate learning rates

HYPOTHESIS: AdamW at 0.0100 for full-context output gates and 0.0115 for pre-handoff gates will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Split output gates by attention role, lowering only the two full-context gates to 0.0100 while preserving 0.0115 for the two pre-handoff gates.

EVIDENCE: Joint 0.0115 achieved 0.991177, while lowering only pre-handoff gates to 0.0100 regressed to 0.991917; the complementary split directly tests whether the full-context gates are instead responsible for the shared-rate compromise.

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
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.023, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=full_context_output_gate_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=pre_handoff_output_gate_params, lr=scalar_lr * 0.023, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE