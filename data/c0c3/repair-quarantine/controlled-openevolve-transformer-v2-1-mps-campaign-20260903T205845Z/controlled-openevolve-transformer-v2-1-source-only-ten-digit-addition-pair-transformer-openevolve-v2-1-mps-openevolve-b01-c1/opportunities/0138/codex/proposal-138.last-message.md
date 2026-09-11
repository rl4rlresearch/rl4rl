MECHANISM: Scale-aware orthonormal key-row LayerNorm quotient

HYPOTHESIS: Expressing the first key row in an orthonormal basis of the centered pre-affine LayerNorm space will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because it removes the same unobservable coefficient as the failed zero-coordinate quotient without its scale-dependent conditioning defect.

INTENDED_EDIT: Replace the first dense key-projection row with seven learned tangent-space coordinates, reconstruct its effective weight using the current `ln1` scales, and preserve the original initialization draws and all other QKV parameters.

EVIDENCE: The single-key-row zero-coordinate quotient fell to 62.02% despite being representationally redundant on centered LayerNorm states. This directly motivates testing an orthonormal, dynamically scale-aware parameterization that preserves the quotient’s full observable function class while avoiding a privileged omitted coordinate.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class OneTangentKeyRowQKV(nn.Linear):
    """QKV map with one key row parameterized on the LayerNorm tangent space."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()

        basis = torch.zeros(d_model, d_model - 1)
        for j in range(d_model - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("key_basis", basis, persistent=False)

        self.first_key_row = nn.Parameter(
            (full_weight[d_model] @ basis).clone()
        )
        self.weight = nn.Parameter(
            torch.cat(
                (full_weight[:d_model], full_weight[d_model + 1 :]), dim=0
            ).clone()
        )
        self.bias = nn.Parameter(
            torch.cat(
                (full_bias[:d_model], full_bias[2 * d_model :])
            ).clone()
        )

    def forward(
        self, x: torch.Tensor, ln_scale: torch.Tensor
    ) -> torch.Tensor:
        d_model = self.in_features
        first_key_weight = (
            self.key_basis @ self.first_key_row
        ) / ln_scale
        full_weight = torch.cat(
            (
                self.weight[:d_model],
                first_key_weight.unsqueeze(0),
                self.weight[d_model:],
            ),
            dim=0,
        )
        full_bias = torch.cat(
            (
                self.bias[:d_model],
                self.bias.new_zeros(d_model),
                self.bias[d_model:],
            )
        )
        return F.linear(x, full_weight, full_bias)


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
        # Parameterize one key row directly on the centered LayerNorm space.
        # Its omitted normal component is constant across key positions.
        self.qkv = OneTangentKeyRowQKV(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)
=======
    def forward(
        self, x: torch.Tensor, ln1_scale: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x, ln1_scale)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(x)
        x = x + self.attn(normalized, self.ln1.weight)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeResidualLinear):
=======
                nn.init.zeros_(module.bias)
        elif isinstance(module, OneTangentKeyRowQKV):
            # Draw the original dense QKV matrix, then project the first key
            # row orthogonally off the LayerNorm-normal direction.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                d_model = module.in_features
                module.weight.copy_(
                    torch.cat(
                        (full[:d_model], full[d_model + 1 :]), dim=0
                    )
                )
                module.first_key_row.copy_(
                    full[d_model] @ module.key_basis
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeResidualLinear):
>>>>>>> REPLACE