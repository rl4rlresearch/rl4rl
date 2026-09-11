MECHANISM: Spectral relative-distance routing

HYPOTHESIS: Replacing the 30-parameter partially shared distance lookup with 20 learned head-specific DCT coefficients will reduce the model from 606 to 596 parameters while retaining at least 99% accuracy, because ten smooth modes per head preserve flexible causal routing without constraining the fragile lexical or absolute-position representations.

INTENDED_EDIT: Parameterize each attention head’s gauge-fixed relative bias as a learned ten-mode cosine expansion over the currently active distance range, while retaining the eleven fixed-zero long-distance endpoints.

EVIDENCE: The 606-parameter model reached 0.9989 accuracy with extensive sharing and fixed endpoints in its relative-bias table, indicating that unconstrained per-distance biases are not all essential. The 603-parameter fixed-spectrum positional experiment fell to 0.565, so this patch leaves all four independently learned absolute-position amplitudes and the lexical path unchanged and instead tests a different, fully learned spectral routing mechanism.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next nine
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 23))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twenty_first = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twentieth = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_nineteenth = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_eighteenth = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_preantepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_antepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Learn routing in a compact cosine basis over the active distances;
        # retain the proven fixed-zero window at the eleven longest distances.
        self.relative_bias_spectrum = nn.Parameter(torch.zeros(n_head, 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_twenty_second = (
            self.relative_bias_core_twenty_second.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twenty_first = (
            self.relative_bias_core_twenty_first.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twentieth = (
            self.relative_bias_core_twentieth.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_nineteenth = (
            self.relative_bias_core_nineteenth.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_eighteenth = (
            self.relative_bias_core_eighteenth.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_preantepenultimate = (
            self.relative_bias_core_preantepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_antepenultimate = (
            self.relative_bias_core_antepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_penultimate = (
            self.relative_bias_core_penultimate.expand(self.n_head).unsqueeze(1)
        )
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
                self.relative_bias_endpoint,
            )
        ).unsqueeze(1)
        relative_bias_endpoint = torch.cat(
            (
                self.relative_bias_endpoint,
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
            )
        ).unsqueeze(1)
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_second,
                relative_bias_core_twenty_first,
                relative_bias_core_twentieth,
                relative_bias_core_nineteenth,
                relative_bias_core_eighteenth,
                relative_bias_core_preantepenultimate,
                relative_bias_core_antepenultimate,
                relative_bias_core_penultimate,
                relative_bias_core_endpoint,
                relative_bias_penultimate,
                relative_bias_endpoint,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
=======
        active_distance_count = self.mask.size(0) - 11
        distance_values = torch.arange(
            active_distance_count,
            device=x.device,
            dtype=x.dtype,
        )
        frequencies = torch.arange(
            1,
            self.relative_bias_spectrum.size(1) + 1,
            device=x.device,
            dtype=x.dtype,
        )
        relative_features = torch.cos(
            math.pi
            * (distance_values.unsqueeze(1) + 0.5)
            * frequencies.unsqueeze(0)
            / active_distance_count
        )
        relative_features = (
            relative_features - relative_features[:1]
        ) * math.sqrt(2.0 / active_distance_count)
        learned_relative_bias = (
            self.relative_bias_spectrum @ relative_features.T
        )
        relative_bias = torch.cat(
            (
                learned_relative_bias,
                learned_relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
>>>>>>> REPLACE