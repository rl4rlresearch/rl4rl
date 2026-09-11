MECHANISM: Mean-zero terminal MLP bias gauge

HYPOTHESIS: Constraining the terminal MLP output bias to the seven-dimensional mean-zero subspace will retain at least 99% accuracy with 1,627 parameters, because its omitted all-ones component is exactly erased by the immediately following final LayerNorm under the fixed one-block, zero-dropout configuration.

INTENDED_EDIT: Apply the verified key-bias and `ln2`-bias removals, then replace `fc2` with a linear layer whose eight-dimensional bias is reconstructed from seven learned orthonormal mean-zero coordinates.

EVIDENCE: The 1,628-parameter reference achieved 99.95% accuracy. Unlike the failed attention-output-bias removal, this removes only a scalar offset immediately canceled by final LayerNorm; retaining all seven functional `fc2` bias directions also avoids the capacity loss tested by broader bias ablations.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class MeanZeroBiasLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        # Preserve the original Linear construction's RNG consumption.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_coords = nn.Parameter(torch.zeros(out_features - 1))

        basis = torch.zeros(out_features, out_features - 1)
        for j in range(out_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_basis @ self.bias_coords
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroBiasLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE