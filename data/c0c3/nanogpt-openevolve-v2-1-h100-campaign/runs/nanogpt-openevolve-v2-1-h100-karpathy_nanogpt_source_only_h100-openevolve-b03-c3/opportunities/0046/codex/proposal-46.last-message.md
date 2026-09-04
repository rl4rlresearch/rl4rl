MECHANISM: Locally refined adaptive output-gate optimization

HYPOTHESIS: AdamW at 0.012 for only the four output-gate matrices on the proven 704/1024/1344 backbone will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Restore the best staged-attention schedule, exclude output-gate matrices from Muon, and optimize them with dedicated AdamW at 0.012.

EVIDENCE: Output-gate AdamW at 0.0115 achieved the best observed val_bpb of 0.991177, versus 0.991514 at 0.010 and 0.991889 at 0.015; local quadratic interpolation of these three results places the estimated minimum near 0.0121.

<<<<<<< SEARCH
        # Preserve the original total short-attention budget while shifting
        # context toward the representation handed to each full-context layer.
        short_windows = (
            3 * long_window // 8,
            long_window // 2,
            5 * long_window // 8,
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
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.024, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE