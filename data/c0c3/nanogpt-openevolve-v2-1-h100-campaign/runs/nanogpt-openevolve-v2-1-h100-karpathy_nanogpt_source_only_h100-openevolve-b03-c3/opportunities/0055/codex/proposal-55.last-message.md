MECHANISM: Adaptive optimization of staged-attention output gates

HYPOTHESIS: Moving only the four output-gate matrices to AdamW at 0.0115 on the current 768/1024/1280 schedule will lower val_bpb below 0.991835 while retaining at least 470M training tokens.

INTENDED_EDIT: Exclude handoff and full-context output gates from Muon and place them in a dedicated AdamW parameter group without changing the current attention allocation.

EVIDENCE: Dedicated output-gate AdamW at 0.0115 produced the best observed val_bpb of 0.991177 on the 704/1024/1344 schedule; applying it to the current schedule isolates whether that optimizer improvement generalizes across short-window allocations.

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
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.023, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE