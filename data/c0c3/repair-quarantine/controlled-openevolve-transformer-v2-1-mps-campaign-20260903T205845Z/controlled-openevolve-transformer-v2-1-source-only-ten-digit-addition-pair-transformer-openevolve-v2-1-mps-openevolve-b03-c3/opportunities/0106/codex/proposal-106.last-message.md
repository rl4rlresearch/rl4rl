MECHANISM: Asymmetric third query-bias coordinate pruning

HYPOTHESIS: A 1,375-parameter transformer will retain at least 99% accuracy because the verified 1,376-parameter design achieved 99.61%, while fixing one additional initially zero query-bias coordinate preserves all projection weights, value paths, and relative-lag routing.

INTENDED_EDIT: Replace the eight-coordinate query bias with five learned coordinates, reconstructing the final coordinate of both heads and the penultimate coordinate of the first head as zero.

EVIDENCE: Balanced per-head query-bias pruning reached 99.61% at 1,376 parameters, whereas reductions involving projection, scale, or MLP gauges failed; extending the successful bias-pruning mechanism by one scalar is the closest informative reduction.

<<<<<<< SEARCH
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        # Fix both per-head terminal coordinates and one additional
        # initially zero coordinate in the first head.
        self.q_bias = nn.Parameter(
            torch.zeros(d_model - n_head - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)
=======
        bsz, seqlen, d_model = x.shape
        first_learned = self.q_bias[: self.head_dim - 2]
        second_learned = self.q_bias[self.head_dim - 2 :]
        q_bias = torch.cat(
            (
                first_learned,
                self.q_bias.new_zeros(2),
                second_learned,
                self.q_bias.new_zeros(1),
            )
        )
        q = self.q_proj(x) + q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)
>>>>>>> REPLACE