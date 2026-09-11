MECHANISM: Complementary signed-lag attention

HYPOTHESIS: A single unrestricted discrete lag potential, used with opposite signs by the two heads, will retain at least 99% accuracy while reducing the current model from 1,315 to 1,285 parameters because the heads primarily need complementary operand routes rather than independent lag tables.

INTENDED_EDIT: Replace the two independent 30-parameter lag tables with one gauge-fixed 30-parameter table whose positive and negative logits define complementary learned causal routes.

EVIDENCE: Dense stationary lag routing reached 99.97%, while the Gaussian mixture reached only 49.85%; this suggests discrete per-lag freedom is load-bearing, but does not establish that the two heads require independent tables. The signed table preserves unrestricted discrete resolution and direct optimization while testing head-route independence.

<<<<<<< SEARCH
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
=======
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_head != 2:
            raise ValueError("complementary lag routing requires two heads")

        self.n_head = n_head
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Each head learns a stationary causal routing preference. The omitted
        # final lag fixes the softmax-invariant common shift of each table.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
=======
        # One unrestricted discrete potential defines complementary routes:
        # one head seeks its maxima and the other its minima. The omitted final
        # lag fixes the remaining softmax-invariant scalar shift.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = lag_bias[:, lag].unsqueeze(0)
=======
        base_lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(1),
            ),
            dim=-1,
        )
        lag_bias = torch.stack(
            (base_lag_bias, -base_lag_bias),
            dim=0,
        )
        att = lag_bias[:, lag].unsqueeze(0)
>>>>>>> REPLACE