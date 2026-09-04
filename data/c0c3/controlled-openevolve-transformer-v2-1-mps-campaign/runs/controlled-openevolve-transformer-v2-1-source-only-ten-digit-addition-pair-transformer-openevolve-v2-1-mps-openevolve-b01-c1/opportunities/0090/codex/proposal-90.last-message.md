MECHANISM: Complementary head-anchored bilinear attention

HYPOTHESIS: Gauge-fixing each head’s query coordinate system while preserving its independent learned attention-score function will reduce the model from 1,538 to 1,506 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Replace the redundant learned query/key coordinate frames with head-specific anchored query maps. Canonically transform fresh query initialization into complementary fixed coordinate blocks and compensate in each learned key map, preserving initial attention scores and all independent value pathways.

EVIDENCE: Sharing one query projection across heads collapsed accuracy to 0.01%, showing that head-specific score functions are load-bearing. This patch retains separate query tails, keys, biases, and values for every head; it removes only the 16-parameter query/key basis gauge per head and preserves the freshly initialized score functions.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class HeadAnchoredQKVLinear(nn.Linear):
    """QKV map with a fixed coordinate chart for each head's query/key product."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.anchor_scale = 0.02

        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        query_free, key_weight, query_bias = self._canonicalize(
            full_weight, full_bias[:d_model]
        )
        value_weight = full_weight[2 * d_model :].clone()
        value_bias = full_bias[2 * d_model :].clone()

        del self.weight
        del self.bias
        self.query_free = nn.Parameter(query_free.clone())
        self.key_weight = nn.Parameter(key_weight.clone())
        self.value_weight = nn.Parameter(value_weight)
        self.query_bias = nn.Parameter(query_bias.clone())
        self.value_bias = nn.Parameter(value_bias)
        self.register_buffer(
            "query_anchor",
            self.anchor_scale
            * torch.eye(
                self.head_dim,
                device=full_weight.device,
                dtype=full_weight.dtype,
            ),
            persistent=False,
        )

    def _canonicalize(
        self, full_weight: torch.Tensor, query_bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_free = []
        key_weight = []
        transformed_bias = []

        for head in range(self.n_head):
            row_start = head * self.head_dim
            row_end = row_start + self.head_dim
            anchor_start = row_start
            anchor_end = anchor_start + self.head_dim

            query = full_weight[row_start:row_end]
            key = full_weight[
                self.d_model + row_start : self.d_model + row_end
            ]
            anchor = query[:, anchor_start:anchor_end]

            canonical_query = self.anchor_scale * torch.linalg.solve(
                anchor, query
            )
            canonical_key = (
                anchor.transpose(0, 1) / self.anchor_scale
            ) @ key
            canonical_bias = self.anchor_scale * torch.linalg.solve(
                anchor, query_bias[row_start:row_end]
            )

            query_free.append(
                torch.cat(
                    (
                        canonical_query[:, :anchor_start],
                        canonical_query[:, anchor_end:],
                    ),
                    dim=1,
                )
            )
            key_weight.append(canonical_key)
            transformed_bias.append(canonical_bias)

        return (
            torch.stack(query_free, dim=0),
            torch.cat(key_weight, dim=0),
            torch.cat(transformed_bias, dim=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_weight = []
        for head in range(self.n_head):
            anchor_start = head * self.head_dim
            free = self.query_free[head]
            query_weight.append(
                torch.cat(
                    (
                        free[:, :anchor_start],
                        self.query_anchor,
                        free[:, anchor_start:],
                    ),
                    dim=1,
                )
            )

        query_weight = torch.stack(query_weight, dim=0).reshape(
            self.d_model, self.d_model
        )
        query = F.linear(x, query_weight, self.query_bias)
        key = F.linear(x, self.key_weight)
        value = F.linear(x, self.value_weight, self.value_bias)
        return torch.cat((query, key, value), dim=-1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        self.qkv = HeadAnchoredQKVLinear(d_model, n_head)
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
        elif isinstance(module, DistributedEightPrunedInputWeightLinear):
=======
        elif isinstance(module, HeadAnchoredQKVLinear):
            # Draw the original full QKV matrix, then quotient each head's
            # query/key basis while preserving its initial attention scores.
            with torch.no_grad():
                full = torch.empty(
                    3 * module.d_model,
                    module.d_model,
                    device=module.query_free.device,
                    dtype=module.query_free.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                query_free, key_weight, query_bias = module._canonicalize(
                    full, full.new_zeros(module.d_model)
                )
                module.query_free.copy_(query_free)
                module.key_weight.copy_(key_weight)
                module.value_weight.copy_(full[2 * module.d_model :])
                module.query_bias.copy_(query_bias)
                module.value_bias.zero_()
        elif isinstance(module, DistributedEightPrunedInputWeightLinear):
>>>>>>> REPLACE