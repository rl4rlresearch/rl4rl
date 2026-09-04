MECHANISM: Third per-position residual common-mode quotient

HYPOTHESIS: Zero-centering positional row 2 will reduce the verified 1,610-parameter model to 1,609 parameters while retaining at least 99% accuracy, because its removed uniform component is canceled locally by pre-attention LayerNorm and ultimately by the final LayerNorm.

INTENDED_EDIT: Represent positional rows 1 and 2 with independent seven-dimensional Helmert coordinates, reconstruct both as zero-mean vectors, and retain full parameters only from row 3 onward.

EVIDENCE: Zero-centering positional row 1 produced the current qualified 1,610-parameter model at 99.6% accuracy, while the analogous positional-origin common-mode quotient achieved 99.92%; applying the same verified residual invariance to the next positional row is the smallest directly supported reduction.

<<<<<<< SEARCH
        centered_position = embedding.weight[1] - embedding.weight[1].mean()
        self.position_weight = nn.Parameter(
            (position_basis.transpose(0, 1) @ centered_position).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[2:].detach().clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.origin_basis @ self.origin_weight
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
        second_position = self.position_basis @ self.position_weight
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                second_position.unsqueeze(0),
                self.weight,
            ),
            dim=0,
        )
=======
        centered_positions = embedding.weight[1:3] - embedding.weight[1:3].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[3:].detach().clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.origin_basis @ self.origin_weight
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
        compact_positions = (
            self.position_weight @ self.position_basis.transpose(0, 1)
        )
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                compact_positions,
                self.weight,
            ),
            dim=0,
        )
>>>>>>> REPLACE