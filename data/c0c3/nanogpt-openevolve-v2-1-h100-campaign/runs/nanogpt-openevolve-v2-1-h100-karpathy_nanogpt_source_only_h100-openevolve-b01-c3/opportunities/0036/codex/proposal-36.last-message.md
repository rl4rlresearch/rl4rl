MECHANISM: Final-layer-only global attention reallocation

HYPOTHESIS: Restoring the verified-best 5× MLP while removing the unsuccessful bigram expert and making the intermediate full-attention layer local will retain final-layer global context, raise throughput above 460M tokens, and reduce val_bpb below 0.982905.

INTENDED_EDIT: Remove the rank-64 bigram path, widen all MLPs to 5×, and change the attention pattern from SSSL to SSSS; the existing window logic still forces the final layer to use full context.

EVIDENCE: The unbiased 5× MLP achieved the best observed val_bpb of 0.982905 despite processing 14.7M fewer tokens than the 4.375× baseline, while the bigram expert regressed to 0.985446 and cost 9.4M tokens; reclaiming one intermediate full-attention layer tests whether extra token exposure can improve the proven 5× design without eliminating global attention.

<<<<<<< SEARCH
    n_embd: int = 768
    window_pattern: str = "SSSL"
    bigram_rank: int = 64
=======
    n_embd: int = 768
    window_pattern: str = "SSSS"
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_dim = 35 * config.n_embd // 8
=======
        hidden_dim = 5 * config.n_embd
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Low-rank transition expert: logits[next_token | current_token].
        self.bigram_embed = nn.Embedding(config.vocab_size, config.bigram_rank)
        self.bigram_head = nn.Linear(config.bigram_rank, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
>>>>>>> REPLACE

<<<<<<< SEARCH
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.bigram_embed.weight, mean=0.0, std=1.0)
        # Preserve the baseline function initially; the expert learns a residual.
        torch.nn.init.zeros_(self.bigram_head.weight)
=======
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        self.bigram_embed.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
=======
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.bigram_embed.weight.numel() +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        bigram = sum(p.numel() for p in self.bigram_embed.parameters())
        bigram += sum(p.numel() for p in self.bigram_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + bigram + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'bigram': bigram, 'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }
=======
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        bigram_embed_params = list(self.bigram_embed.parameters())
        bigram_head_params = list(self.bigram_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(bigram_embed_params) + len(bigram_head_params) +
            len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) +
            len(resid_params) + len(x0_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
        param_groups = [
            dict(kind='adamw', params=lm_head_params + bigram_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params + bigram_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
=======
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        bigram_logits = self.bigram_head(norm(self.bigram_embed(idx)))
        logits = (logits + bigram_logits).float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # final layer is still forced to full context
>>>>>>> REPLACE