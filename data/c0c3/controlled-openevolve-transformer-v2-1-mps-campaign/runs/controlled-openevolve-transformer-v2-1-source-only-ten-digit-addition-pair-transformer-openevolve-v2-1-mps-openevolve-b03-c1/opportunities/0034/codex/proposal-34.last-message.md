MECHANISM: Isolated attention-output LayerNorm gauge fixing

HYPOTHESIS: Constraining only the attention output projection to the zero-mean feature subspace will reduce the model from 1,478 to 1,470 parameters while retaining at least 99% accuracy, because its removed per-token all-ones component is erased by subsequent LayerNorms under the fixed zero-dropout configuration.

INTENDED_EDIT: Store the 8-by-8 attention output projection as 7-by-8 coefficients in a fixed orthonormal zero-mean basis, reconstruct its centered 8-dimensional output during forward passes, and preserve the observable initialized function.

EVIDENCE: The combined attention/MLP output reduction failed at 84.22%, while the 1,478-parameter baseline achieved 99.95%; isolating the smaller eight-parameter attention component directly tests which part of that combined change caused the optimization failure.

<<<<<<< SEARCH
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Distance zero fixes each head's softmax-invariant additive gauge;
=======
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Any all-ones component emitted into the residual stream is removed
        # by subsequent LayerNorms. Parameterize the observable zero-mean
        # output subspace with an orthonormal basis.
        basis = torch.eye(d_model)[:, : d_model - 1]
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.register_buffer("proj_basis", basis, persistent=False)

        # Distance zero fixes each head's softmax-invariant additive gauge;
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
                (self.proj_basis.T @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_weight = self.proj_basis @ self.proj.weight
        y = F.linear(y, proj_weight)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.mlp.gauge_fix_fc1()
=======
        for block in self.blocks:
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
>>>>>>> REPLACE