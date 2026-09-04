MECHANISM: Residual-stream LayerNorm shift gauge fixing

HYPOTHESIS: Constraining the attention projection and MLP output to the zero-mean feature subspace will reduce parameters from 1,478 to 1,457 while retaining at least 99% accuracy, because their removed all-ones output components only add per-token scalar shifts that all subsequent LayerNorms erase under the fixed zero-dropout configuration.

INTENDED_EDIT: Store the attention output projection as 7-by-8 coefficients and the MLP output weight and bias as 7-dimensional coefficients in fixed orthonormal zero-mean bases, reconstructing their centered 8-dimensional forms during forward passes.

EVIDENCE: Centering the harmonic positional readout along the same LayerNorm-null all-ones direction achieved 99.98% accuracy, and orthonormally removing the LayerNorm-null components of `fc1` retained 99.95% accuracy at 1,478 parameters.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        # Preserve construction order while removing the bias representable by
        # the retained value bias through this projection.
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
        # Preserve construction order while removing the bias representable by
        # the retained value bias through this projection.
        self.proj.bias = None

        # A component shared by every residual-stream coordinate is removed by
        # subsequent LayerNorms. Represent projection outputs in an orthonormal
        # basis for the complementary zero-mean subspace.
        basis = torch.eye(d_model)[:, : d_model - 1]
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.register_buffer("output_basis", basis, persistent=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        self.register_buffer("mask", mask, persistent=False)

    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.output_basis.T @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_weight = self.output_basis @ self.proj.weight
        y = F.linear(y, proj_weight)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            bias = self.fc2.bias
            centered_weight = weight - weight.mean(dim=0, keepdim=True)
            centered_bias = bias - bias.mean()
            self.fc2.weight = nn.Parameter(
                (self.fc1_basis @ centered_weight).clone()
            )
            self.fc2.bias = nn.Parameter(
                (self.fc1_basis @ centered_bias).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.mlp.gauge_fix_fc1()
=======
        for block in self.blocks:
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
            block.mlp.gauge_fix_fc2()
>>>>>>> REPLACE