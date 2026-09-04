MECHANISM: Five-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,599-parameter design by quotienting `fc2` column 5 will yield 1,598 parameters with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while the optimization-sensitive column 3 remains unrestricted.

INTENDED_EDIT: Compact `fc2` columns 0, 1, 2, 4, and 5, reconstruct their original ordering in the forward pass, and train the compact matrix with full-coordinate AdamW moments.

EVIDENCE: Reference Design 1 achieved 99.46% accuracy at 1,599 parameters with columns 0, 1, 2, and 4 compacted, whereas using column 3 as the fourth quotient collapsed to 95.54%; adding adjacent column 5 is the closest incremental reduction that preserves the successful unrestricted treatment of column 3.

<<<<<<< SEARCH
    def compact_output_column(self) -> None:
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first column in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_column = self.bias_basis.transpose(0, 1) @ weight[:, 0]
            remaining_weight = weight[:, 1:].clone()
        self.fc2_first_column = nn.Parameter(compact_column)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        first_column = (
            self.bias_basis @ self.fc2_first_column
        ).unsqueeze(1)
        weight = torch.cat((first_column, self.fc2.weight), dim=1)
        return self.drop(F.linear(hidden, weight, bias))
=======
    def compact_output_columns(self) -> None:
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four and five,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 6:]), dim=1
            ).clone()
        self.fc2_compact_columns = nn.Parameter(compact_columns)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        compact_columns = self.bias_basis @ self.fc2_compact_columns
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight[:, :1],
                compact_columns[:, 3:],
                self.fc2.weight[:, 1:],
            ),
            dim=1,
        )
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.mlp.compact_output_column()
=======
            block.mlp.compact_output_columns()
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.mlp.fc2_first_column, blk.mlp.bias_basis),
=======
            (blk.mlp.fc2_compact_columns, blk.mlp.bias_basis),
>>>>>>> REPLACE