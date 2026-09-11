MECHANISM: Hybrid learned-absolute and Fourier-relative position encoding

HYPOTHESIS: Replacing two of six arbitrary positional lookup channels with a fixed mean-free Fourier pair and two learned amplitudes will remove `2 * (max_seq_len - 2)` parameters while retaining at least 99% accuracy, because four learned channels preserve absolute-position flexibility while the Fourier pair supplies a translation-compatible routing signal.

INTENDED_EDIT: Retain four learned mean-free positional profiles, replace the fifth and sixth profiles with analytic sine/cosine features, and initialize their learned amplitudes to match the discarded channels’ RMS scale.

EVIDENCE: The 1,524-parameter design reached 99.98% after all positional common modes were removed, while recent one-parameter LayerNorm and attention reductions repeatedly failed. This suggests testing the load-bearing assumption that every remaining positional channel requires an independent lookup table rather than continuing along the exhausted scalar-quotient path.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with all position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_first_common", torch.empty(()), persistent=False
        )
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fifth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_third_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fifth_common", torch.empty(()), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
        for j in range(embedding_dim - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        position_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            position_basis[: j + 1, j] = 1.0 / scale
            position_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("position_basis", position_basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first_weight = (self.position_basis @ self.weight).unsqueeze(1)
        remaining = F.embedding(idx, first_weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
        second = F.embedding(idx, second_weight)
        third_weight = (
            self.position_basis @ self.third_coordinate
        ).unsqueeze(1)
        third = F.embedding(idx, third_weight)
        fourth_weight = (
            self.position_basis @ self.fourth_coordinate
        ).unsqueeze(1)
        fourth = F.embedding(idx, fourth_weight)
        fifth_weight = (
            self.position_basis @ self.fifth_coordinate
        ).unsqueeze(1)
        fifth = F.embedding(idx, fifth_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (
                remaining,
                second,
                third,
                fourth,
                fifth,
                last,
            ),
            dim=-1,
        )
        return coordinates @ self.basis.transpose(0, 1)
=======
class MeanFreePositionEmbedding(nn.Module):
    """Four learned absolute coordinates plus a scaled Fourier pair."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_first_common", torch.empty(()), persistent=False
        )
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourier_scale = nn.Parameter(torch.empty(2))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_third_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fifth_common", torch.zeros(()), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
        for j in range(embedding_dim - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        position_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            position_basis[: j + 1, j] = 1.0 / scale
            position_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("position_basis", position_basis, persistent=False)

        phase = (
            2.0
            * math.pi
            * torch.arange(num_embeddings, dtype=torch.float32)
            / num_embeddings
        )
        fourier_features = math.sqrt(2.0) * torch.stack(
            (torch.sin(phase), torch.cos(phase)), dim=1
        )
        fourier_features.sub_(fourier_features.mean(dim=0, keepdim=True))
        self.register_buffer(
            "fourier_features", fourier_features, persistent=False
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first_weight = (self.position_basis @ self.weight).unsqueeze(1)
        remaining = F.embedding(idx, first_weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
        second = F.embedding(idx, second_weight)
        third_weight = (
            self.position_basis @ self.third_coordinate
        ).unsqueeze(1)
        third = F.embedding(idx, third_weight)
        fourth_weight = (
            self.position_basis @ self.fourth_coordinate
        ).unsqueeze(1)
        fourth = F.embedding(idx, fourth_weight)
        fourier = F.embedding(idx, self.fourier_features)
        fourier = fourier * self.fourier_scale
        coordinates = torch.cat(
            (
                remaining,
                second,
                third,
                fourth,
                fourier,
            ),
            dim=-1,
        )
        return coordinates @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from all
            # retained content coordinates.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, 0] @ module.position_basis)
                module.removed_first_common.copy_(projected[:, 0].mean())
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.fourth_coordinate.copy_(
                    projected[:, 3] @ module.position_basis
                )
                module.fifth_coordinate.copy_(
                    projected[:, 4] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
                module.removed_third_common.copy_(projected[:, 2].mean())
                module.removed_fourth_common.copy_(projected[:, 3].mean())
                module.removed_fifth_common.copy_(projected[:, 4].mean())
=======
        elif isinstance(module, MeanFreePositionEmbedding):
            # Preserve the original full-width draw, initialize four learned
            # mean-free lookup coordinates, and give the Fourier pair the RMS
            # scale of the two lookup channels it replaces.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, 0] @ module.position_basis)
                module.removed_first_common.copy_(projected[:, 0].mean())
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.fourth_coordinate.copy_(
                    projected[:, 3] @ module.position_basis
                )
                module.fourier_scale.copy_(
                    projected[:, 4:].square().mean(dim=0).sqrt()
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
                module.removed_third_common.copy_(projected[:, 2].mean())
                module.removed_fourth_common.copy_(projected[:, 3].mean())
                module.removed_fifth_common.zero_()
>>>>>>> REPLACE