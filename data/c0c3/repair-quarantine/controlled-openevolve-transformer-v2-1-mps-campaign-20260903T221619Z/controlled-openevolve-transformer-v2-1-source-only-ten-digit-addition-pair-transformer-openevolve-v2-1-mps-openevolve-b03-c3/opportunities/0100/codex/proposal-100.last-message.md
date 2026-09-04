MECHANISM: Asymmetric query-key/value head widths

HYPOTHESIS: A 1,356-parameter transformer will retain at least 99% accuracy because three learned value coordinates per head can transport digit information while the full four-dimensional query-key paths and dense learned relative-lag routing preserve the content-addressing mechanism shown to be load-bearing.

INTENDED_EDIT: Reproduce the qualified all-column terminal gauge, then challenge the assumption that attention values require the same width as query-key vectors by narrowing only the shared learned value stream from four to three dimensions and projecting the resulting six head outputs back into the eight-dimensional residual stream.

EVIDENCE: The 1,380-parameter all-terminal-gauge reference reached 99.71%, while the 1,270-parameter content-independent design collapsed after removing query-key addressing. This motivates retaining full query-key and relative-lag computation while testing a narrower value-transport mechanism.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and ten output-shift column gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(10)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 10)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and every output-shift gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(in_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 10:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.bias.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
=======
        weight = torch.stack(full_weight_prefix, dim=1)
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        # One additive constant per head is omitted because softmax is
        # invariant to shifting all valid relative-lag logits equally.
        self.relative_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 1))
                for _ in range(n_head)
            ]
        )
        self.full_relative_bias = None
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Addressing retains the full head dimension, while a narrower learned
        # value stream transports content through each independently routed
        # attention map.
        self.value_dim = max(1, self.head_dim - 1)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        # One additive constant per head is omitted because softmax is
        # invariant to shifting all valid relative-lag logits equally.
        self.relative_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 1))
                for _ in range(n_head)
            ]
        )
        self.full_relative_bias = None
        self.proj = GaugeFixedAttentionProjection(
            n_head * self.value_dim, d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(
            bsz, seqlen, self.n_head * self.value_dim
        )
        y = self.proj(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column]
                        - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 10:])
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.bias.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column]
                        - raw_weight[-1, column]
                    )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE