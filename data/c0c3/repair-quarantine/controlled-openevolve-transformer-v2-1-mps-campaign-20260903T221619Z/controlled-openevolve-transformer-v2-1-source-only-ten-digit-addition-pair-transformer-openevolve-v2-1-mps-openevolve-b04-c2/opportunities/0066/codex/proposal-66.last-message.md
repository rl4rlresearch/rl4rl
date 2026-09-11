MECHANISM: Alternate-column MLP residual-output quotient

HYPOTHESIS: Quotienting `fc2` columns 0, 1, 2, and 4 will produce a 1,599-parameter model with at least 99% accuracy, avoiding the optimization collapse observed when column 3 was the fourth quotient.

INTENDED_EDIT: Extend the qualified three-column quotient with original `fc2` column 4, preserve column ordering during reconstruction, and leave column 3 unrestricted.

EVIDENCE: Quotienting columns 0–2 achieved 99.90% at 1,600 parameters, while adding adjacent column 3 fell to 95.54%; reallocating the same exact LayerNorm-null quotient to column 4 directly tests whether that failure was specific to column 3’s optimization path.

<<<<<<< SEARCH
    def compact_output_columns(self) -> None:
        # A uniform component of any output-weight column produces only a
        # per-token common-mode residual shift, which the final LayerNorm
        # erases. Store the first two columns in the zero-sum output basis.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1) @ weight[:, :2]
            )
            remaining_weight = weight[:, 2:].clone()
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
        return self.drop(F.linear(hidden, weight, bias))
>>>>>>> REPLACE