MECHANISM: Final key-channel scale gauge fixing

HYPOTHESIS: The resulting 668-parameter transformer will retain at least 99% accuracy because the qualified 669-parameter design reached 99.97%, and fixing its already-shared first coefficient at the same 0.02 anchor used by every sibling key channel removes a scale degree of freedom without tying another input-feature direction.

INTENDED_EDIT: Reproduce the qualified two-coordinate key sharing and single query-bias fixation, while replacing the first shared key coefficient with fixed 0.02 anchors in both heads and retaining the second coefficient as learned and shared.

EVIDENCE: Reference Design 3 achieved 99.97% with two learned cross-head key ties and one fixed query-bias coordinate, whereas tying the third input coefficient collapsed accuracy; fixing the first coefficient follows the existing successful per-channel key-scale normalization instead of constraining that failed third feature direction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Gauge-fixed QKV map sharing one learned key feature across heads."""
=======
class GaugeFixedQKV(nn.Module):
    """Gauge-fixed QKV map sharing one learned key feature across heads."""
>>>>>>> REPLACE

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
        # Reuse the first coefficient of the final key channel from the
        # first head in the corresponding channel of the second head.
        # Queries and all other key coordinates remain head-specific.
        tied_source_index = (
            d_model + head_dim - 1
        ) * in_features
        tied_index = (
            d_model + 2 * head_dim - 1
        ) * in_features
=======
        # The first coefficient of every key channel is fixed at the same
        # scale anchor. Reuse the second coefficient of the final key
        # channel across heads; all later coordinates remain head-specific.
        tied_source_index = (
            d_model + head_dim - 1
        ) * in_features + 1
        tied_index = (
            d_model + 2 * head_dim - 1
        ) * in_features + 1
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = QuotientOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE