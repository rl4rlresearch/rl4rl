MECHANISM: First-head boundary-bias suffix extension

HYPOTHESIS: A 668-parameter transformer will retain at least 99% accuracy because it preserves the qualified 669-parameter architecture while fixing only the adjacent fifth-longest first-head relative-bias coefficient.

INTENDED_EDIT: Reconstruct the qualified 669-parameter query gauge, four-coordinate MLP bias fixing, and first-head boundary suffix, then extend that suffix by one relative-bias coefficient.

EVIDENCE: Successive first-head boundary reductions achieved 99.45% at 670 parameters and 99.11% at 669, while additional query, lexical, normalization, and MLP-bias constraints failed; this motivates continuing the only reduction mechanism that remained qualified.

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases, shares
        # its three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Distance zero and the two longest distances are fixed for both
        # heads. The first head also fixes its next three longest distances.
        # The second head fixes its next six longest biases, shares its three
        # longest remaining positions, and separately shares the adjacent
        # pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 12)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
=======
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 3
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 3 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                self.relative_bias.new_zeros(3),
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=3
        )
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE