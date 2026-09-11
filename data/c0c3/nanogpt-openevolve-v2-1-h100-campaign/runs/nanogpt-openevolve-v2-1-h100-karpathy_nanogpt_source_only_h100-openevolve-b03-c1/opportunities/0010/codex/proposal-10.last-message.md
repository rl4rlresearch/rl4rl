MECHANISM: Recurrently shared lexical value memory

HYPOTHESIS: Reusing one value-embedding lookup across all value-residual layers will preserve the winning attention geometry while reducing redundant parameters, activations, and optimizer work enough to achieve val_bpb below 0.993365.

INTENDED_EDIT: Replace four independent per-layer value-embedding tables with one shared table, looked up once per sequence and modulated by the existing layer-specific, input-dependent gates.

EVIDENCE: Grouped-query attention reduced parameters but caused a throughput cliff to 421.0M tokens and val_bpb 1.012022 because it changed FlashAttention head geometry. This tests parameter sharing without changing the successful four-query-head SSSL attention kernels.

<<<<<<< SEARCH
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
=======
        # A shared lexical value memory is recurrently injected at alternating layers.
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embed = nn.Embedding(config.vocab_size, kv_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
=======
        # Shared value embedding
        torch.nn.init.uniform_(self.value_embed.weight, -s, s)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
=======
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        self.value_embed.to(dtype=torch.bfloat16)
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
=======
        value_embeds_numel = self.value_embed.weight.numel()
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
=======
        value_embeds = sum(p.numel() for p in self.value_embed.parameters())
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_embeds_params = list(self.value_embeds.parameters())
=======
        value_embeds_params = list(self.value_embed.parameters())
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
=======
        x = self.transformer.wte(idx)
        shared_ve = self.value_embed(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = shared_ve if block.attn.ve_gate is not None else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
>>>>>>> REPLACE