MECHANISM: Tenth-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,594-parameter design by quotienting `fc2` column 10 will produce a 1,593-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.

INTENDED_EDIT: Parameterize `fc2` columns 0, 1, 2, and 4 through 10 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave columns 3 and 11 unrestricted.

EVIDENCE: Quotienting columns 0, 1, 2, and 4 through 9 achieved 99.81% accuracy at 1,594 parameters; extending the successful adjacent run to column 10 is the closest incremental reduction using the same exact LayerNorm-null direction.

<<<<<<< SEARCH
    def compact_output_columns(self) -> None:
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first three columns in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1) @ weight[:, :3]
            )
            remaining_weight = weight[:, 3:].clone()
        self.fc2_compact_columns = nn.Parameter(compact_columns)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        compact_columns = self.bias_basis @ self.fc2_compact_columns
        weight = torch.cat((compact_columns, self.fc2.weight), dim=1)
        return self.drop(F.linear(hidden, weight, bias))
=======
    def compact_output_columns(self) -> None:
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
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE