MECHANISM: Adam-optimized attention-output gating

HYPOTHESIS: Restoring the proven unfused first-32-channel output gates and optimizing only those tiny gate matrices with AdamW at 0.01 will lower val_bpb below 0.993637 while retaining at least 480M trained tokens.

INTENDED_EDIT: Undo the unsuccessful value/output forward fusion, exclude output-gate matrices from Muon, and place them in a dedicated AdamW parameter group.

EVIDENCE: Unfused all-layer output gating achieved the best val_bpb, 0.993637 on 482.3M tokens, while the fused implementation regressed to 0.994532 on 474.0M; conditioning and amplitude changes also regressed, motivating preservation of the successful gate computation while testing whether Muon is ill-suited to its tiny 4x32 matrices.

<<<<<<< SEARCH
        # Compute value and output controls together on value-embedding layers.
        gate_input = x[..., :self.ve_gate_channels]
        if ve is not None:
            gate_logits = F.linear(
                gate_input,
                torch.cat((self.ve_gate.weight, self.out_gate.weight), dim=0),
            )
            ve_gate_logits = gate_logits[..., :self.n_kv_head]
            out_gate_logits = gate_logits[..., self.n_kv_head:]
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + (2 * torch.sigmoid(ve_gate_logits)).unsqueeze(-1) * ve
        else:
            out_gate_logits = self.out_gate(gate_input)
=======
        # Keep value and output gates independent so their gradients and compiled
        # projections match the strongest verified gated design.
        gate_input = x[..., :self.ve_gate_channels]
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            ve_gate = 2 * torch.sigmoid(self.ve_gate(gate_input))
            v = v + ve_gate.unsqueeze(-1) * ve
        out_gate_logits = self.out_gate(gate_input)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
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
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        gate_lr=0.01):
        model_dim = self.config.n_embd
        output_gate_params = [
            p for block in self.transformer.h for p in block.attn.out_gate.parameters()
        ]
        output_gate_ids = {id(p) for p in output_gate_params}
        matrix_params = [
            p for p in self.transformer.h.parameters() if id(p) not in output_gate_ids
        ]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(output_gate_params) +
            len(embedding_params) + len(lm_head_params) + len(value_embeds_params) +
            len(resid_params) + len(x0_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=output_gate_params, lr=gate_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
GATE_LR = 0.01          # learning rate for attention-output gates (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
=======
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    gate_lr=GATE_LR,
    adam_betas=ADAM_BETAS,
>>>>>>> REPLACE