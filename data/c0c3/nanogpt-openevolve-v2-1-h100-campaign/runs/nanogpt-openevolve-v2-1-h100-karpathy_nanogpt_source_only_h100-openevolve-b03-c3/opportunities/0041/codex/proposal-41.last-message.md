MECHANISM: Role-aware adaptive optimization for multiplicative gates

HYPOTHESIS: Optimizing the four sigmoid output-gate matrices with AdamW at 0.01 instead of Muon will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.

INTENDED_EDIT: Separate handoff output-gate weights from transformer matrices and optimize them with the same adaptive learning rate used by the successful embedding-context coefficients.

EVIDENCE: Four handoff output gates improved val_bpb to 0.992110, while subsequent context-window refinements plateaued at 0.991729 or regressed; these small multiplicative controllers are a proven lever but are currently orthogonalized as ordinary feature matrices by Muon.

<<<<<<< SEARCH
        matrix_params = list(self.transformer.h.parameters())
=======
        output_gate_params = [
            block.attn.out_gate.weight for block in self.transformer.h
            if block.attn.out_gate is not None
        ]
        output_gate_ids = {id(p) for p in output_gate_params}
        matrix_params = [
            p for p in self.transformer.h.parameters()
            if id(p) not in output_gate_ids
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(context_params))
=======
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(context_params) + len(output_gate_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
=======
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE