MECHANISM: Six-column MLP residual-output common-mode quotient

HYPOTHESIS: Extending the verified 1,598-parameter design by quotienting `fc2` column 6 will produce a 1,597-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while optimization-sensitive column 3 remains unrestricted.

INTENDED_EDIT: Compact `fc2` columns 0, 1, 2, 4, 5, and 6, reconstruct their original ordering during the forward pass, and optimize the compact matrix with full-coordinate AdamW moments.

EVIDENCE: Quotienting columns 0, 1, 2, 4, and 5 achieved 99.93% accuracy at 1,598 parameters, while the earlier failure involved column 3; extending the successful contiguous run after column 4 to adjacent column 6 is the closest incremental reduction.

<<<<<<< SEARCH
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
=======
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def compact_output_columns(self) -> None:
        # Uniform output components are erased by the final LayerNorm.
        # Quotient columns zero through two plus columns four through six,
        # retaining column three as an unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6)]
            )
            remaining_weight = torch.cat(
                (weight[:, 3:4], weight[:, 7:]), dim=1
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
        for block in self.blocks:
            block.attn.compact_value_basis()
=======
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
=======
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.mlp.fc2_compact_columns, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
>>>>>>> REPLACE