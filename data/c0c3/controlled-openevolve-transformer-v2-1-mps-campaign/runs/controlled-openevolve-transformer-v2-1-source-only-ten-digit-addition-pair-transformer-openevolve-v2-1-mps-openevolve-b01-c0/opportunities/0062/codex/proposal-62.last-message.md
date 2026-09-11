MECHANISM: Shared third-coordinate normalization contrast

HYPOTHESIS: Reusing `attn.qv_bias[2]` for the third-coordinate `ln2` contrast while retaining an independent fifth-coordinate contrast will reduce the model to 1607 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Remove one standalone `ln2` parameter, source its third-coordinate contrast from the parameter already shared by `ln1`, and keep its fifth-coordinate contrast independently learned.

EVIDENCE: Neither third-only nor fifth-only `ln2` met the threshold, indicating both contrasts contribute, while sharing the required third-coordinate `ln1` contrast with `qv_bias[2]` achieved 100% at one fewer parameter; this motivates adaptive sharing of the aligned `ln2` contrast rather than deleting it.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 6))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat(
            (
                self.bias.new_zeros(2),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
                self.bias.new_zeros(2),
            )
        )
=======
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
        self.eps = 1e-5

    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                shared_bias.new_zeros(2),
                shared_bias.reshape(1),
                shared_bias.new_zeros(1),
                self.bias,
                shared_bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x + self.mlp(self.ln2(x))
=======
        x = x + self.mlp(self.ln2(x, self.attn.qv_bias[2]))
>>>>>>> REPLACE