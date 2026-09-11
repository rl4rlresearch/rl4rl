MECHANISM: Multi-depth static lexical ensemble

HYPOTHESIS: A channel-wise mixture of all four existing value embeddings will reduce val_bpb below 0.994364 without materially reducing throughput.

INTENDED_EDIT: Preserve the current final-value lexical path at initialization, while adding zero-initialized channel coefficients that let earlier-layer value embeddings contribute directly to the readout.

EVIDENCE: Channel-wise static lexical scaling achieved the best val_bpb of 0.994364, whereas context gating and a dedicated lexical embedding regressed; reusing already-computed value embeddings extends the successful static mechanism without adding another embedding table.

<<<<<<< SEARCH
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
=======
        # Per-channel mixture of every value embedding in the direct lexical expert.
        # Earlier sources start disabled so initialization matches the best design.
        num_value_embeds = sum(has_ve(i, config.n_layer) for i in range(config.n_layer))
        self.bigram_lambdas = nn.Parameter(torch.zeros(num_value_embeds, config.n_embd))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bigram_lambdas.fill_(0.1)
=======
        self.bigram_lambdas.zero_()
        self.bigram_lambdas[-1].fill_(0.1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])

        # The final layer always has a value embedding. Route it directly to the
        # readout with channel-wise scaling alongside the contextual state.
        lexical = norm(ve)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical)
=======
        x0 = x
        lexical_values = []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if ve is not None:
                lexical_values.append(ve)
            x = block(x, ve, cos_sin, self.window_sizes[i])

        # Mix the distinct lexical features learned by every value-embedding
        # depth using static channel-wise coefficients.
        lexical = norm(lexical_values[0]) * self.bigram_lambdas[0].view(1, 1, -1)
        for i in range(1, len(lexical_values)):
            lexical = lexical + norm(lexical_values[i]) * self.bigram_lambdas[i].view(1, 1, -1)
        x = norm(x + lexical)
>>>>>>> REPLACE