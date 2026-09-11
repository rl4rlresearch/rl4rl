MECHANISM: Dedicated lexical-expert embedding

HYPOTHESIS: Decoupling the successful channel-wise lexical expert from the final attention value embedding will eliminate competing attention/readout gradients and reduce val_bpb below 0.994364 without materially reducing throughput.

INTENDED_EDIT: Add a dedicated token embedding for the lexical readout, initialize it as an exact copy of the final value embedding, and optimize it with the existing embedding schedule.

EVIDENCE: Channel-wise static lexical scaling improved val_bpb from 0.995511 to 0.994364, while context-dependent gating regressed to 0.995708; this motivates greater static lexical specialization while preserving the best model’s initial function.

<<<<<<< SEARCH
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
=======
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Dedicated embedding for the direct lexical prediction expert
        self.bigram_embed = nn.Embedding(config.vocab_size, config.n_embd)
        # Rotary embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
=======
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Preserve the existing initial function while allowing the lexical
        # representation to specialize independently during training.
        self.bigram_embed.weight.copy_(
            self.value_embeds[str(self.config.n_layer - 1)].weight
        )
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
=======
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        self.bigram_embed.to(dtype=torch.bfloat16)
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel())
=======
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.bigram_embed.weight.numel() +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
=======
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        bigram_embed = sum(p.numel() for p in self.bigram_embed.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
        total = wte + value_embeds + bigram_embed + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'bigram_embed': bigram_embed,
            'lm_head': lm_head, 'transformer_matrices': transformer_matrices,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds_params = list(self.value_embeds.parameters())
=======
        value_embeds_params = list(self.value_embeds.parameters()) + list(self.bigram_embed.parameters())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The final layer always has a value embedding. Route it directly to the
        # readout with channel-wise scaling alongside the contextual state.
        lexical = norm(ve)
=======
        # Route a dedicated token representation to the readout with channel-wise
        # scaling, leaving the final attention value embedding free to specialize.
        lexical = norm(self.bigram_embed(idx))
>>>>>>> REPLACE