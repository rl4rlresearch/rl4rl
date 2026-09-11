MECHANISM: Nonadjacent first-position residual-stream common-mode gauge

HYPOTHESIS: Representing positional row 0 in the same seven-dimensional zero-mean basis already used successfully by eight positional rows will reduce the model to 1,574 parameters while retaining at least 99% accuracy, because an all-ones shift at one position is eliminated by every pre-LayerNorm path and the final LayerNorm.

INTENDED_EDIT: Compact positional row 0 from eight weights to seven orthonormal coordinates, reconstruct it during forward passes, and preserve full-draw initialization for RNG alignment.

EVIDENCE: The verified 1,575-parameter design reaches 99.91% while eight positional rows already use this exact basis. The successful nonadjacent final-column `fc2` gauge also motivates testing an isolated coordinate after adjacent extensions proved optimization-sensitive; unlike the failed second token-embedding anchor, this gauge does not alter the tied input/output embedding.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.earlier_gauge_index = self.earlier_gauge_flat_start
=======
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.initial_gauge_index = 0
        self.initial_gauge_end_index = embedding_dim - 1
        self.earlier_gauge_index = (
            self.initial_gauge_end_index
            + self.earlier_gauge_flat_start
            - embedding_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.anchor_index = self.tie_index + embedding_dim - 1

        earlier_gauge_coords = (
=======
        self.anchor_index = self.tie_index + embedding_dim - 1

        initial_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[:embedding_dim]
        )
        earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        compact_weight = torch.cat(
            (
                flat_weight[: self.earlier_gauge_flat_start],
                earlier_gauge_coords,
=======
        compact_weight = torch.cat(
            (
                initial_gauge_coords,
                flat_weight[embedding_dim : self.earlier_gauge_flat_start],
                earlier_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        earlier_gauge_row = (
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        initial_gauge_row = (
            self.gauge_basis
            @ self.weight[
                self.initial_gauge_index : self.initial_gauge_end_index
            ]
        )
        earlier_gauge_row = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        flat_weight = torch.cat(
            (
                self.weight[: self.earlier_gauge_index],
                earlier_gauge_row,
=======
        flat_weight = torch.cat(
            (
                initial_gauge_row,
                self.weight[
                    self.initial_gauge_end_index : self.earlier_gauge_index
                ],
                earlier_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_weight = full_weight.flatten()
            earlier_gauge_coords = (
=======
            flat_weight = full_weight.flatten()
            initial_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[: module.embedding_dim]
            )
            earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_weight = torch.cat(
                (
                    flat_weight[: module.earlier_gauge_flat_start],
                    earlier_gauge_coords,
=======
            compact_weight = torch.cat(
                (
                    initial_gauge_coords,
                    flat_weight[
                        module.embedding_dim : module.earlier_gauge_flat_start
                    ],
                    earlier_gauge_coords,
>>>>>>> REPLACE