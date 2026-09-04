MECHANISM: Extended adjacent relative-bias sharing

HYPOTHESIS: A 669-parameter transformer will retain at least 99% accuracy because the qualified 670-parameter model already shares adjacent second-head routing biases, while the failed third key tie indicates that further constraining the key projection is substantially riskier.

INTENDED_EDIT: Deduplicate one second-head relative-position coefficient by extending its existing two-position shared bias to three adjacent positions, preserving both cross-head key ties and all query and MLP biases.

EVIDENCE: The current 670-parameter design achieved 99.88%, whereas extending key sharing to a third coefficient collapsed accuracy to 0.01%; Reference Design 1 achieved 99.97% with three-way adjacent long-distance bias sharing, motivating this orthogonal reduction.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases, shares
        # its three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and
        # shares each of the two preceding groups of three positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 10)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
=======
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(3),
                second_head_bias[-1:].expand(3),
>>>>>>> REPLACE