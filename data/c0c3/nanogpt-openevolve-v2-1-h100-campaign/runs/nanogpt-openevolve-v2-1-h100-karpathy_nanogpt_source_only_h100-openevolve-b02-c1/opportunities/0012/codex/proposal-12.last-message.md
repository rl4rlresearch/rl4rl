MECHANISM: Context-conditioned channel-wise lexical gating

HYPOTHESIS: Modulating the successful channel-wise lexical residual with a learned context-dependent gate will reduce val_bpb below 0.994364 without materially reducing throughput.

INTENDED_EDIT: Add a zero-initialized 32-to-512 readout gate conditioned on the final contextual state, preserving the current lexical path exactly at initialization while allowing token-feature scaling to vary by context.

EVIDENCE: Expanding lexical scaling from four head-level coefficients to 512 channel-level coefficients improved val_bpb from 0.995511 to 0.994364 at comparable throughput, motivating a minimally expensive increase in the same path’s expressivity.

<<<<<<< SEARCH
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        # Value embeddings
=======
        # Per-channel strength of the direct lexical prediction expert, plus a
        # small context-conditioned gate that is neutral at initialization.
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        self.bigram_gate_channels = 32
        self.bigram_gate = nn.Linear(self.bigram_gate_channels, config.n_embd, bias=False)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bigram_lambdas.fill_(0.1)
        # Value embeddings
=======
        self.bigram_lambdas.fill_(0.1)
        torch.nn.init.zeros_(self.bigram_gate.weight)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
=======
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        readout_gate = sum(p.numel() for p in self.bigram_gate.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + readout_gate + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'readout_gate': readout_gate,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas, self.bigram_lambdas]
        x0_params = [self.x0_lambdas]
=======
        resid_params = [self.resid_lambdas, self.bigram_lambdas, self.bigram_gate.weight]
        x0_params = [self.x0_lambdas]
>>>>>>> REPLACE

<<<<<<< SEARCH
        lexical = norm(ve)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical)
=======
        lexical = norm(ve)
        context_gate = 2 * torch.sigmoid(
            self.bigram_gate(norm(x)[..., :self.bigram_gate_channels])
        )
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1) * context_gate
        x = norm(x + lexical)
>>>>>>> REPLACE