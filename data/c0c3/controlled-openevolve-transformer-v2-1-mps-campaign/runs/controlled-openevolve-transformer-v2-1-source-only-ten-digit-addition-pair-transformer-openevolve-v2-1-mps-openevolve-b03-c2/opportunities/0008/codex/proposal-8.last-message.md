MECHANISM: Orthogonal attention-key and residual-shift gauge removal

HYPOTHESIS: Removing one proven softmax-invariant key-bias coordinate and one uniform MLP-output bias degree will produce a 1642-parameter model with at least 99% accuracy, since the final LayerNorm cancels any feature-uniform shift from `fc2.bias`.

INTENDED_EDIT: Preserve constructor RNG consumption while shortening the fused QKV bias by one element and the MLP output bias by one element, reconstructing fixed-zero gauge coordinates during each projection.

EVIDENCE: The one-coordinate key-bias design achieved 99.88% with 1643 parameters, while removing a second key coordinate fell to 98.52%; this tests a distinct exact invariance instead of further perturbing the sensitive key-bias parameterization.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Fix one softmax-invariant key-bias coordinate at zero.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # A feature-uniform output shift is canceled by the final LayerNorm,
        # so fix one bias coordinate as a gauge choice.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc2_bias = torch.cat((self.fc2.bias.new_zeros(1), self.fc2.bias))
        hidden = F.gelu(self.fc1(x))
        return self.drop(F.linear(hidden, self.fc2.weight, fc2_bias))
>>>>>>> REPLACE