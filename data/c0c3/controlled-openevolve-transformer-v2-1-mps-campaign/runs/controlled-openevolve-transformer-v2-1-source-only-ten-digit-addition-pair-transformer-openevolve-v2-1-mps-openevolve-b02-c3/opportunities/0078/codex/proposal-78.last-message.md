MECHANISM: First-head key-scale gauge completion with maximum-distance bias sharing

HYPOTHESIS: An 821-parameter transformer will retain at least 99% accuracy because sharing the maximum-distance bias preserves the qualified 822-parameter design, while fixing the remaining first-head key-channel scale uses a reciprocal Q/K basis degree without tying the functionally important adjacent-distance biases.

INTENDED_EDIT: Reconstruct the qualified shared maximum-distance relative bias and fix the third first-head key channel’s leading coefficient at 0.02, reducing the current 823-parameter model by two parameters.

EVIDENCE: Maximum-distance sharing achieved 99.94% at 822 parameters, whereas extending sharing to the adjacent distance collapsed to 5.46%; the successful 99.98% fixing of all three first-head value-channel scales motivates testing the analogous remaining scale constraint on the first-head key basis instead.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        selected_indices = {
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with one completed first-head key scale gauge."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        selected_key_channels = {
            head * head_dim + offset
            for head in range(n_head)
            for offset in range(min(2, head_dim))
        }
        if head_dim > 2:
            selected_key_channels.add(2)
        selected_indices = {
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)

        # Keep every causal distance learned while sharing only the uniquely
        # sparse maximum-distance bias across heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
        self.farthest_relative_bias = nn.Parameter(torch.zeros(1))

        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias[:, : seqlen - 1],
            ),
            dim=1,
        )
        att = att + relative_bias[:, distance.clamp_min(0)].unsqueeze(0)
=======
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.farthest_relative_bias.expand(self.n_head, 1),
            ),
            dim=1,
        )
        relative_bias = torch.cat(
            (
                learned_relative_bias.new_zeros(self.n_head, 1),
                learned_relative_bias[:, : seqlen - 1],
            ),
            dim=1,
        )
        att = att + relative_bias[:, distance.clamp_min(0)].unsqueeze(0)
>>>>>>> REPLACE