MECHANISM: Complete per-head Q/K scale gauge with final-only validation

HYPOTHESIS: A 662-parameter transformer will achieve at least 99% accuracy because fixing the last key-channel scale preserves the 663-parameter model’s representable attention scores, while final-only validation removes the runtime bottleneck that prevented both prior 662-parameter attempts from finishing.

INTENDED_EDIT: Reconstruct the qualified first-head-quintet design, fix the remaining second-head key scale at 0.02, and evaluate only the final trained state so both required checkpoints are written within the time limit.

EVIDENCE: The analogous first-head gauge fixing achieved 99.88% at 663 parameters; both second-head attempts timed out without producing negative accuracy evidence, so reducing validation overhead directly tests the same exact attention symmetry.

<<<<<<< SEARCH
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
=======
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(3, head_dim))
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent triplet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 14)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quintet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 16)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 5
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 5 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(3),
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 7
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 7 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(5),
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE