MECHANISM: Low-rank causal prefix memory

HYPOTHESIS: Replacing token-conditioned lag scaling with a learned 32-dimensional summary of all preceding embeddings, while restoring the best fixed two-lag FIR and four handoff gates, will lower val_bpb below 0.992110 while retaining at least 465M trained tokens.

INTENDED_EDIT: Restore fixed two-lag embedding mixing and add a neutral-initialized low-rank nonlinear projection of the causal running-mean embedding history before the transformer.

EVIDENCE: Fixed two-lag mixing with four handoff gates achieved 0.992110, while a third lag and token-conditioned lag gates regressed to 0.992455 and 0.992360. This challenges the shared assumption that additional cheap context must be another local lag: the quarter-context regression shows broader history matters, so a compressed global prefix memory tests a distinct recurrent-style context path.

<<<<<<< SEARCH
        # Channel-wise coefficients for embeddings one and two tokens back,
        # with token-conditioned selection of the useful local-context lag.
        self.embedding_context = nn.Parameter(torch.zeros(2, config.n_embd))
        self.embedding_context_gate = nn.Linear(32, 2, bias=False)
=======
        # Proven fixed two-lag local path plus a distinct compressed global
        # memory of all preceding token embeddings.
        self.embedding_context = nn.Parameter(torch.zeros(2, config.n_embd))
        self.context_memory_down = nn.Linear(config.n_embd, 32, bias=False)
        self.context_memory_up = nn.Linear(32, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.embedding_context.zero_()
        torch.nn.init.zeros_(self.embedding_context_gate.weight)
        # Value embeddings
=======
        self.embedding_context.zero_()
        torch.nn.init.uniform_(self.context_memory_down.weight, -s, s)
        torch.nn.init.zeros_(self.context_memory_up.weight)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        context_mix = self.embedding_context.numel() + self.embedding_context_gate.weight.numel()
=======
        context_mix = (self.embedding_context.numel() +
                       sum(p.numel() for p in self.context_memory_down.parameters()) +
                       sum(p.numel() for p in self.context_memory_up.parameters()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        context_params = [self.embedding_context, self.embedding_context_gate.weight]
=======
        context_params = ([self.embedding_context] +
                          list(self.context_memory_down.parameters()) +
                          list(self.context_memory_up.parameters()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.transformer.wte(idx)
        # Token-conditioned depthwise causal FIR supplies selected local context.
        # Left padding prevents information crossing independent sequence boundaries.
        padded_x = F.pad(x, (0, 0, 2, 0))
        previous_x = padded_x[:, 1:-1]
        previous2_x = padded_x[:, :-2]
        context_mix = torch.tanh(self.embedding_context).to(dtype=x.dtype)
        context_gate = 2 * torch.sigmoid(self.embedding_context_gate(x[..., :32]))
        x = (x +
             context_gate[..., 0, None] * context_mix[0] * previous_x +
             context_gate[..., 1, None] * context_mix[1] * previous2_x)
        x = norm(x)
=======
        x = self.transformer.wte(idx)
        # A causal running mean exposes the entire preceding history through a
        # learned low-rank nonlinear memory path. The zero-initialized up
        # projection makes this path neutral at initialization.
        prefix_sum = x.cumsum(dim=1) - x
        prefix_count = torch.arange(
            T, device=x.device, dtype=torch.float32
        ).clamp_min(1.0)
        prefix_mean = prefix_sum / prefix_count.to(dtype=x.dtype)[None, :, None]
        context_memory = self.context_memory_up(
            F.silu(self.context_memory_down(norm(prefix_mean)))
        )

        # Retain the best fixed depthwise two-lag local-context path.
        # Left padding prevents information crossing independent sequences.
        padded_x = F.pad(x, (0, 0, 2, 0))
        previous_x = padded_x[:, 1:-1]
        previous2_x = padded_x[:, :-2]
        context_mix = torch.tanh(self.embedding_context).to(dtype=x.dtype)
        x = (x + context_mix[0] * previous_x +
             context_mix[1] * previous2_x)
        x = norm(x + context_memory)
>>>>>>> REPLACE