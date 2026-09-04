MECHANISM: Ninth-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,595-parameter design by quotienting `fc2` column 9 will produce a 1,594-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.

INTENDED_EDIT: Parameterize `fc2` columns 0, 1, 2, and 4 through 9 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave column 3 and columns 10–11 unrestricted.

EVIDENCE: Quotienting through column 8 achieved 99.97% accuracy at 1,595 parameters; extending the same exact LayerNorm-null quotient to adjacent column 9 is the closest incremental reduction.

<<<<<<< SEARCH
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus column four, while retaining
        # column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 5:]), dim=1
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
                compact_columns[:, 3:4],
                self.fc2.weight[:, 1:],
            ),
            dim=1,
        )
=======
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through nine,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 10:]), dim=1
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
>>>>>>> REPLACE