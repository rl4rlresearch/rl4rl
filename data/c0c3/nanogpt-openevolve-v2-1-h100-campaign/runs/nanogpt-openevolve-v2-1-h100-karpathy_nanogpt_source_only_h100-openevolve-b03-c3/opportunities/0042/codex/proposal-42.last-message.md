MECHANISM: Role-aware adaptive optimization for all multiplicative gates

HYPOTHESIS: Extending the successful AdamW treatment from output gates to value-residual gates will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

INTENDED_EDIT: Restore the best 704/1024/1344 staged-attention backbone and optimize all eight sigmoid gate matrices with AdamW at 0.01 instead of Muon.

EVIDENCE: On the 704/1024/1344 design, moving output-gate matrices from Muon to AdamW improved val_bpb from 0.991729 to 0.991514; value-residual gates are equally small multiplicative controllers and remain optimized by Muon.

<<<<<<< SEARCH
        # Preserve the original total short-attention budget while shifting
        # context toward the representation handed to each full-context layer.
        short_windows = (
            3 * long_window // 8,
            long_window // 2,
            5 * long_window // 8,
        )
=======
        # Preserve total short-attention FLOPs with the strongest verified
        # staged-window allocation.
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
        sigmoid_gate_params = [
            gate.weight
            for block in self.transformer.h
            for gate in (block.attn.ve_gate, block.attn.out_gate)
            if gate is not None
        ]
        sigmoid_gate_ids = {id(p) for p in sigmoid_gate_params}
        matrix_params = [
            p for p in self.transformer.h.parameters()
            if id(p) not in sigmoid_gate_ids
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
            len(x0_params) + len(context_params) + len(sigmoid_gate_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
=======
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=sigmoid_gate_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE