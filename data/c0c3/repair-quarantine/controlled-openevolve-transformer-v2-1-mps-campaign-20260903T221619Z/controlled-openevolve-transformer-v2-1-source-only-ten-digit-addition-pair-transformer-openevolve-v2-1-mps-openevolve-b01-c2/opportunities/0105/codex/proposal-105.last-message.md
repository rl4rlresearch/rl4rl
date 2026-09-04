MECHANISM: Single-position Helmert common-mode quotient

HYPOTHESIS: Representing positional row 1 in a seven-dimensional zero-mean Helmert basis will reduce the verified 1,533-parameter model to 1,532 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove the per-position scalar null direction from positional row 1, preserving the four verified position-zero anchors and their token-embedding transfer.

EVIDENCE: The current Helmert query-row design reaches 99.78% at 1,533 parameters, and the prior Helmert quotient rescued query row 6 to 99.89%; a common feature offset in one positional row is erased by every pre-norm and the final LayerNorm, making this a distinct exact quotient with an already successful coordinate geometry.

<<<<<<< SEARCH
class FourCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 4
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(4), persistent=False
        )
=======
class FourCoordinateGaugedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding. In addition
        # to the four position-zero anchors, quotient the common feature offset
        # of positional row 1.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * embedding_dim - 5
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.register_buffer(
            "_init_token_shift", base.weight.new_zeros(4), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for column in range(embedding_dim - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("position_row_basis", basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:4],
                self.weight.new_zeros(1),
                self.weight[4:],
            )
        )
        return F.embedding(
            indices, flat.view(self.num_embeddings, self.embedding_dim)
        )
=======
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        anchored_row = torch.cat(
            (
                self.weight.new_zeros(2),
                self.weight[:1],
                self.weight.new_zeros(1),
                self.weight[1:4],
                self.weight.new_zeros(1),
            )
        )
        orthonormal_row = (
            self.weight[4:11] @ self.position_row_basis.transpose(0, 1)
        )
        remaining_rows = self.weight[11:].view(
            self.num_embeddings - 2, self.embedding_dim
        )
        dense = torch.cat(
            (
                anchored_row.unsqueeze(0),
                orthonormal_row.unsqueeze(0),
                remaining_rows,
            ),
            dim=0,
        )
        return F.embedding(indices, dense)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FourCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 1], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 7].sub_(shifts[3])
                flat = full.flatten()
                module.weight.copy_(
                    torch.cat((flat[2:3], flat[4:7], flat[8:]))
                )
                module._init_token_shift.copy_(shifts)
=======
        elif isinstance(module, FourCoordinateGaugedPositionEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                shifts = torch.stack(
                    (full[0, 0], full[0, 1], full[0, 3], full[0, 7])
                ).clone()
                full[:, 0].sub_(shifts[0])
                full[:, 1].sub_(shifts[1])
                full[:, 3].sub_(shifts[2])
                full[:, 7].sub_(shifts[3])
                anchored_row = full[0, [2, 4, 5, 6]]
                orthonormal_row = full[1] @ module.position_row_basis
                module.weight.copy_(
                    torch.cat(
                        (
                            anchored_row,
                            orthonormal_row,
                            full[2:].flatten(),
                        )
                    )
                )
                module._init_token_shift.copy_(shifts)
>>>>>>> REPLACE