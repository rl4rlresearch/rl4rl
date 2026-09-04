MECHANISM: Within-head rotational gauge fixing for tied queries and keys

HYPOTHESIS: Removing six rotationally redundant query/key parameters will reduce the model from 1,272 to 1,266 parameters while preserving at least 99% accuracy, because an orthogonal rotation of each head’s mean-zero query coordinates preserves every tied query-key dot product.

INTENDED_EDIT: Replace the dense tied query/key projection with an equivalent Helmert-basis chart whose three-dimensional centered component is upper triangular over its first three input coordinates; retain an unrestricted common component and value projection.

EVIDENCE: The 1,272-parameter design achieved 100% accuracy after an exact embedding gauge reduction. Here, the supplied model uses identical queries and keys and adds the attention-bias scalar equally to every head coordinate, leaving a three-dimensional orthogonal symmetry per four-dimensional head and therefore three removable parameters per head.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
=======
class GaugeFixedQKV(nn.Module):
    """Tied query/key and value map with within-head rotations gauge-fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.in_features = d_model - 1
        self.center_dim = self.head_dim - 1
        if self.center_dim < 1 or self.in_features < self.center_dim:
            raise ValueError("head dimensions do not support the gauge-fixed chart")

        self.q_common = nn.Parameter(torch.empty(n_head, self.in_features))
        self.value_weight = nn.Parameter(torch.empty(d_model, self.in_features))
        nn.init.normal_(
            self.q_common,
            mean=0.0,
            std=0.02 / math.sqrt(self.head_dim),
        )
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)

        self.centered_rows = nn.ParameterList()
        for _ in range(n_head):
            raw = torch.empty(self.center_dim, self.in_features)
            nn.init.normal_(raw, mean=0.0, std=0.02)
            rotation, _ = torch.linalg.qr(raw[:, : self.center_dim])
            gauged = rotation.transpose(0, 1) @ raw
            for row in range(self.center_dim):
                self.centered_rows.append(
                    nn.Parameter(gauged[row, row:].clone())
                )

        basis = torch.zeros(self.head_dim, self.center_dim)
        for col in range(self.center_dim):
            width = col + 1
            scale = math.sqrt(width * (width + 1))
            basis[:width, col] = 1.0 / scale
            basis[width, col] = -width / scale
        self.register_buffer("center_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_chart = x[..., :-1]
        common = F.linear(x_chart, self.q_common)

        heads = []
        offset = 0
        for _ in range(self.n_head):
            rows = []
            for row in range(self.center_dim):
                rows.append(F.pad(self.centered_rows[offset], (row, 0)))
                offset += 1
            heads.append(torch.stack(rows))
        centered_weight = torch.stack(heads)

        centered = torch.einsum(
            "bti,hci->bthc", x_chart, centered_weight
        )
        q = common.unsqueeze(-1) + torch.einsum(
            "bthc,dc->bthd", centered, self.center_basis
        )
        q = q.flatten(-2)
        v = F.linear(x_chart, self.value_weight)
        return torch.cat((q, v), dim=-1)


class FixedBiasLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 2 * d_model, bias=False)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
>>>>>>> REPLACE