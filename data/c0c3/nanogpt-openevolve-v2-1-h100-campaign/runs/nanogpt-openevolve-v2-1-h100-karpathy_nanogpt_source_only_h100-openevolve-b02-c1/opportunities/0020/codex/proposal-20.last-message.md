MECHANISM: Ordered low-rank trigram readout

HYPOTHESIS: Adding a rank-512 multiplicative feature of the previous and current token embeddings will reduce val_bpb below 0.987386 while retaining nearly all throughput.

INTENDED_EDIT: Replace the assumption that non-current lexical context must pass through attention with an asymmetric Hadamard-product expert that maps each ordered previous/current token pair directly into the pre-logit state; initialize its channel coefficients to zero and retain the proven 256K-token update regime.

EVIDENCE: The static channel-wise current-token expert improved val_bpb from 0.995511 to 0.994364, whereas the additive multi-depth ensemble regressed to 0.997915. This motivates extending the successful direct readout with a genuinely new multiplicative token-pair interaction rather than another additive lexical source.

<<<<<<< SEARCH
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        # Value embeddings
=======
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        # Zero initialization preserves the baseline while this learns an ordered,
        # low-rank (previous token, current token) -> next-token interaction.
        self.trigram_lambdas = nn.Parameter(torch.zeros(config.n_embd))
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bigram_lambdas.fill_(0.1)
        # Value embeddings
=======
        self.bigram_lambdas.fill_(0.1)
        self.trigram_lambdas.zero_()
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel())
=======
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.bigram_lambdas.numel() + self.trigram_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel())
=======
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.bigram_lambdas.numel() + self.trigram_lambdas.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas, self.bigram_lambdas]
=======
        resid_params = [self.resid_lambdas, self.bigram_lambdas, self.trigram_lambdas]
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The final layer always has a value embedding. Route it directly to the
        # readout with channel-wise scaling alongside the contextual state.
        lexical = norm(ve)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical)
=======
        # The final value embedding retains the successful current-token expert.
        current_lexical = norm(ve)
        lexical = current_lexical * self.bigram_lambdas.view(1, 1, -1)

        # Directly represent the ordered local token pair. Distinct current-token
        # and previous-token embedding tables make the product asymmetric, while
        # its CP-style channel factorization adds only one learned vector.
        previous_lexical = torch.cat(
            (torch.zeros_like(x0[:, :1]), x0[:, :-1]), dim=1
        )
        token_pair = norm(current_lexical * previous_lexical)
        token_pair = token_pair * self.trigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical + token_pair)
>>>>>>> REPLACE