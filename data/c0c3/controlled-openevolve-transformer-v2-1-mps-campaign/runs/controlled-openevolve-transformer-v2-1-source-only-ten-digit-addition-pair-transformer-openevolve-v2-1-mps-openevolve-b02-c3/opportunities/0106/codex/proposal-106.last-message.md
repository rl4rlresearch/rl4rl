MECHANISM: Asymmetric tolerant-head key-coordinate anchoring

HYPOTHESIS: A 678-parameter transformer will retain at least 99% accuracy because the qualified 679-parameter relative-bias layout is preserved while only the second, demonstrably more tolerant attention head receives the remaining key-coordinate gauge anchor.

INTENDED_EDIT: Reconstruct the qualified 679-parameter six-bias pruning layout, then fix the second head’s third key-channel anchor at 0.02 to remove one additional learned QKV coefficient.

EVIDENCE: The 679-parameter design achieved 99.94%, and repeated asymmetric relative-bias pruning showed the second head tolerates constraints that the first head does not; the failed simultaneous two-head anchor test reached 31.51%, so isolating the anchor to the tolerant head tests whether that failure was caused by constraining both heads together.

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
            for offset in range(
                min(3 if head == n_head - 1 else 2, head_dim)
            )
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next four longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 4)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next six longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = F.pad(self.relative_bias, (0, 4)).view(
            self.n_head, -1
        )
=======
        relative_bias_coeff = F.pad(self.relative_bias, (0, 6)).view(
            self.n_head, -1
        )
>>>>>>> REPLACE