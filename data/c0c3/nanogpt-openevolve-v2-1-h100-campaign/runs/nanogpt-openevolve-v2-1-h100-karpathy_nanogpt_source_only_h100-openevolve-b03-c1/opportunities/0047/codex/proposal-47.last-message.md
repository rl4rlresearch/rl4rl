MECHANISM: Adam-optimized low-dimensional value routing

HYPOTHESIS: Moving the existing 32-channel value gates from Muon to AdamW will reduce val_bpb below 0.984083 by allowing the tiny 4×32 routing matrices to learn correlated per-head updates without Muon’s orthogonalization constraint.

INTENDED_EDIT: Exclude value-gate weights from the Muon matrix groups and optimize them with AdamW at one-quarter of the matrix learning rate, preserving the architecture and all other settings.

EVIDENCE: Learned 32-channel gates achieved 0.984083 versus 0.986944 with fixed gates, while widening their inputs to 64 or 512 channels regressed to 0.984472 and 0.984493. This indicates the compact gates are useful and motivates improving their optimization rather than increasing their capacity.

<<<<<<< SEARCH
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        model_dim = self.config.n_embd
        ve_gate_params = [
            block.attn.ve_gate.weight for block in self.transformer.h
            if block.attn.ve_gate is not None
        ]
        ve_gate_param_ids = {id(p) for p in ve_gate_params}
        matrix_params = [
            p for p in self.transformer.h.parameters()
            if id(p) not in ve_gate_param_ids
        ]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(ve_gate_params) +
            len(embedding_params) + len(lm_head_params) + len(value_embeds_params) +
            len(resid_params) + len(x0_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
=======
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ve_gate_params, lr=matrix_lr * 0.25, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE