MECHANISM: First-head boundary-bias quartet sharing

HYPOTHESIS: A 665-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary triplet sharing to four adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.

INTENDED_EDIT: Reconstruct the qualified 666-parameter design, then share one coefficient across four first-head biases immediately preceding its fixed three-bias suffix.

EVIDENCE: The adaptive triplet-sharing design achieved 99.58% at 666 parameters, whereas fixing the neighboring bias at zero achieved only 98.62%; extending learned sharing is therefore the most informative reduction.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head also fixes its next three longest distances.
        # The second head fixes its next six longest biases, shares its three
        # longest remaining positions, and separately shares the adjacent
        # pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 12)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quartet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 15)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 6
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 6 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(4),
                self.relative_bias.new_zeros(3),
                second_head_bias[:-2],
>>>>>>> REPLACE