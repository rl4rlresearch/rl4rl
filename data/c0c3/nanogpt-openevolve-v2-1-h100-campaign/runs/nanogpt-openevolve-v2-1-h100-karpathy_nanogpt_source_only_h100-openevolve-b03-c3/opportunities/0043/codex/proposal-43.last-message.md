MECHANISM: Faster adaptive learning for handoff output gates

HYPOTHESIS: Optimizing only the four output-gate matrices with AdamW at 0.015 on the proven 704/1024/1344 backbone will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

INTENDED_EDIT: Restore the best staged-attention schedule, exclude output gates from Muon, and raise their dedicated AdamW learning rate from the previously successful 0.01 to 0.015.

EVIDENCE: Moving output gates to AdamW at 0.01 improved val_bpb from 0.991729 to 0.991514, whereas moving all gate matrices to AdamW regressed to 0.992169; this motivates a focused output-gate learning-rate refinement without changing value-gate optimization.

<<<<<<< SEARCH
        # Preserve total short-attention FLOPs while testing the midpoint
        # between the best progression and the overly steep progression.
        short_windows = (
            5 * long_window // 16,
            long_window // 2,
            11 * long_window // 16,
        )
=======
        # Preserve total short-attention FLOPs with the best verified
        # progression toward each full-context handoff.
        short_windows = (
            11 * long_window // 32,
            long_window // 2,
            21 * long_window // 32,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
=======
        model_dim = self.config.n_embd
        output_gate_params = [
            block.attn.out_gate.weight for block in self.transformer.h
            if block.attn.out_gate is not None
        ]
        output_gate_ids = {id(p) for p in output_gate_params}
        matrix_params = [
            p for p in self.transformer.h.parameters()
            if id(p) not in output_gate_ids
        ]
        value_embeds_params = list(self.value_embeds.parameters())
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
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.03, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE