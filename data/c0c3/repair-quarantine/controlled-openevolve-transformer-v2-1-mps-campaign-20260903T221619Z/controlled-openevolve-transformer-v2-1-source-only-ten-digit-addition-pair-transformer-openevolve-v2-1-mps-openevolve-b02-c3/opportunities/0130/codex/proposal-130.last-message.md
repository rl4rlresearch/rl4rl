MECHANISM: Shared-key multi-head content addressing

HYPOTHESIS: A 656-parameter transformer will retain at least 99% accuracy because both heads can use one learned key representation while preserving their load-bearing independent queries, value channels, and positional routing biases.

INTENDED_EDIT: Replace the two independent three-dimensional key projections with one learned three-dimensional key projection shared by both heads, retaining head-specific queries and values and fixing the shared key’s three coordinate scales.

EVIDENCE: The 669-parameter design reached 99.70% after further first-head key gauge fixing, indicating redundancy in key representation, while additional query-bias and relative-bias constraints collapsed accuracy; this motivates compressing keys across heads without disturbing the fragile query coordinates or head-specific positional biases.

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
            (d_model + channel) * in_features
            for channel in selected_key_channels
        }
        selected_indices.update(
            (2 * d_model + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)
=======
class SharedKeyQKV(nn.Module):
    """Head-specific queries and values over one learned shared key space."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        out_features = 3 * d_model
        head_dim = d_model // n_head
        compact_out_features = 2 * d_model + head_dim
        shared_key_start = d_model
        value_start = d_model + head_dim

        # A common key space needs only one coordinate gauge, while each
        # head retains an independent query map and value payload.
        selected_indices = {
            (shared_key_start + offset) * in_features
            for offset in range(min(3, head_dim))
        }
        selected_indices.update(
            (value_start + offset) * in_features
            for offset in range(min(3, head_dim))
        )
        shear_indices = {
            value_start * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (value_start + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
=======
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim
        self.in_features = in_features
        self.out_features = out_features
        self.compact_out_features = compact_out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(
                compact_out_features * in_features
                - len(self.fixed_indices)
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(pieces).view(
            self.out_features, self.in_features
        )
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(quotient_x, weight)
=======
        weight = torch.cat(pieces).view(
            self.compact_out_features, self.in_features
        )
        quotient_x = F.linear(x, self.basis.transpose(0, 1))
        compact_qkv = F.linear(quotient_x, weight)
        q, shared_k, v = torch.split(
            compact_qkv,
            (self.d_model, self.head_dim, self.d_model),
            dim=-1,
        )
        k = torch.cat((shared_k,) * self.n_head, dim=-1)
        return torch.cat((q, k, v), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model, n_head)
=======
        self.qkv = SharedKeyQKV(d_model, n_head)
>>>>>>> REPLACE