MECHANISM: Distributed LayerNorm-tangent QKV projections

HYPOTHESIS: Removing one LayerNorm-normal-direction coefficient from each of the 24 independent QKV rows will reduce the model from 1,525 to 1,501 parameters while retaining at least 99% accuracy, because each projection row has only seven observable linear degrees of freedom on centered LayerNorm states.

INTENDED_EDIT: Replace the assumption that attention requires dense QKV rows with learned row-specific tangent-space projections, distributing omitted coordinates evenly while preserving every head-specific query, key, and value row.

EVIDENCE: The verified 1,525-parameter model reaches 99.88% while applying the same distributed one-coordinate quotient to all twelve `fc1` rows. Unlike key-weight sharing, which collapsed accuracy to 21.68%, this removes no head-specific addressing functions: query/value constants remain representable by their biases, and key constants are softmax-null.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class DistributedLayerNormQuotientQKVLinear(nn.Linear):
    """QKV map with one LayerNorm-normal coefficient removed per row."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 3 * in_features:
            raise ValueError("QKV output width must be three times its input width")
        super().__init__(in_features, out_features)

        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        omitted_coordinate = torch.arange(out_features) % in_features
        keep_mask = torch.ones(
            out_features, in_features, dtype=torch.bool
        )
        keep_mask[
            torch.arange(out_features), omitted_coordinate
        ] = False

        self.weight = nn.Parameter(
            full_weight[keep_mask]
            .view(out_features, in_features - 1)
            .clone()
        )
        self.bias = nn.Parameter(
            torch.cat(
                (
                    full_bias[:in_features],
                    full_bias[2 * in_features :],
                )
            ).clone()
        )
        self.register_buffer(
            "omitted_coordinate", omitted_coordinate, persistent=False
        )
        self.register_buffer("keep_mask", keep_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.weight.new_zeros(
            self.out_features, self.in_features
        )
        full_weight = full_weight.masked_scatter(
            self.keep_mask, self.weight.reshape(-1)
        )

        q_bias = self.bias[: self.in_features]
        v_bias = self.bias[self.in_features :]
        full_bias = torch.cat(
            (
                q_bias,
                q_bias.new_zeros(self.in_features),
                v_bias,
            )
        )
        return F.linear(x, full_weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Preserve independent QKV rows while representing each on the
        # seven-dimensional tangent space of centered LayerNorm outputs.
        self.qkv = DistributedLayerNormQuotientQKVLinear(
            d_model, 3 * d_model
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q, k, v = self.qkv(x).chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeTokenEmbedding):
=======
        elif isinstance(module, DistributedLayerNormQuotientQKVLinear):
            # Draw the original dense QKV matrix to preserve the RNG sequence.
            # At the initial unit LayerNorm scale, subtracting each row's
            # omitted coefficient preserves its action on centered inputs.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                omitted = full.gather(
                    1, module.omitted_coordinate.unsqueeze(1)
                )
                canonical = full - omitted
                module.weight.copy_(
                    canonical[module.keep_mask].view_as(module.weight)
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeTokenEmbedding):
>>>>>>> REPLACE