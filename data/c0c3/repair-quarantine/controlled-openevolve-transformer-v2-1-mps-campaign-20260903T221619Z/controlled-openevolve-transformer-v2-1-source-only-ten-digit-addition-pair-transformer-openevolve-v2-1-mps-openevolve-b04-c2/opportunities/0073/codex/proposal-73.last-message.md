MECHANISM: Final-column MLP residual-output common-mode quotient

HYPOTHESIS: Quotienting `fc2` column 11 will reduce the model from 1,593 to 1,592 learned parameters while retaining at least 99% accuracy, because its uniform residual-channel component is erased by the final LayerNorm and optimization-sensitive column 3 remains unrestricted.

INTENDED_EDIT: Parameterize every `fc2` output column except column 3 in the seven-dimensional zero-sum basis, reconstruct the original ordering in the forward pass, and continue using full-coordinate AdamW moments.

EVIDENCE: The current design achieved 99.69% accuracy after quotienting columns 0, 1, 2, and 4 through 10; extending the same exact function-null quotient to the sole remaining adjacent column 11 is the closest incremental reduction.

<<<<<<< SEARCH
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through ten,
        # retaining columns three and eleven as unrestricted coordinates.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9, 10)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 11:]), dim=1
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
=======
        # Uniform output components are erased by the final LayerNorm.
        # Quotient every column except column three, which remains an
        # unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11)]
            )
            remaining_weight = weight[:, 3:4].clone()
        self.fc2_compact_columns = nn.Parameter(compact_columns)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        compact_columns = self.bias_basis @ self.fc2_compact_columns
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
>>>>>>> REPLACE