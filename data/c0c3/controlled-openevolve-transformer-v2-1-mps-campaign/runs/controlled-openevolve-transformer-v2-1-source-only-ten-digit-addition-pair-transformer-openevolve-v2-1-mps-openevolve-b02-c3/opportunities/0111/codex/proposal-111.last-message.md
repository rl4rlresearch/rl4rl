MECHANISM: Second-head value/output scale-gauge fixing

HYPOTHESIS: A 676-parameter transformer will retain at least 99% accuracy because the qualified three-way relative-bias sharing is preserved while one second-head value-channel scale is fixed at 0.02 and remains compensable by the learned output projection.

INTENDED_EDIT: Reconstruct the qualified 677-parameter three-way second-head bias sharing, then remove one learned QKV coefficient by applying the existing first-head value-scale gauge convention to one second-head value channel.

EVIDENCE: The 677-parameter three-way-sharing design achieved 99.97%; the source already successfully fixes the corresponding value scales for all first-head channels, while the failed 673-parameter trial changed the entire value transport, motivating this isolated one-coordinate gauge reduction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""
=======
class GaugeFixedQKV(nn.Module):
    """QKV map also fixing one second-head value/output scale gauge."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        shear_indices = {
=======
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        selected_indices.add(
            (2 * d_model + head_dim) * in_features
        )
        shear_indices = {
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next six longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 6)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = F.pad(self.relative_bias, (0, 6)).view(
            self.n_head, -1
        )
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE