MECHANISM: Value-bias absorption into attention output bias

HYPOTHESIS: Removing the seven remaining value-bias coordinates will reduce parameters from 1,586 to 1,579 while retaining at least 99% accuracy, because normalized attention adds the same value-bias vector at every position and its mean-free projected effect can be represented exactly by the learned attention output bias.

INTENDED_EDIT: Retain only the eight query-bias parameters in QKV and reconstruct both key and value biases as zeros, preserving the original constructor RNG consumption and all projection weights.

EVIDENCE: Exact removal of all key-bias coordinates achieved 99.94% at 1,586 parameters, while the mean-free attention projection retains seven learned output-bias directions; this supports eliminating the remaining algebraically redundant attention bias rather than pruning load-bearing LayerNorm capacity.

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
=======
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and the value bias absorbed by the learned output-projection bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(full_bias[:d_model].clone())
        self.proj = MeanFreeResidualLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        full_bias = torch.cat((q_bias, q_bias.new_zeros(2 * d_model)))
>>>>>>> REPLACE