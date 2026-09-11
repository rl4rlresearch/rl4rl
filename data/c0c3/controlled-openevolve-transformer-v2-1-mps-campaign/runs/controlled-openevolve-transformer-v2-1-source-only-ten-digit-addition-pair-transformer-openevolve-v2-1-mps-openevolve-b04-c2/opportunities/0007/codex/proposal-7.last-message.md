MECHANISM: Gauge-fixed terminal MLP bias plus softmax-invariant key-bias elimination

HYPOTHESIS: Building on the verified 1,636-parameter key-bias design, fixing one common-mode degree of freedom in the terminal MLP bias will reduce the model to 1,635 parameters while retaining at least 99% accuracy, because the immediately following final LayerNorm exactly cancels uniform shifts across the eight residual channels.

INTENDED_EDIT: Apply the verified initialization- and optimizer-order-preserving key-bias removal, then store seven terminal MLP output-bias coordinates and reconstruct an eight-dimensional bias with the final coordinate fixed to zero.

EVIDENCE: The optimizer-order-preserving key-bias design achieved 99.71% accuracy with 1,636 parameters, whereas larger or optimization-disrupting reductions failed; this motivates removing only one additional algebraically redundant parameter while preserving widths and all weight matrices.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve the original parameter slot and initialization RNG use while
        # omitting the key bias, which cancels within each attention softmax row.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
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
        # The final LayerNorm removes any common shift of this bias, so choose
        # the equivalent representative whose last channel is fixed at zero.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
>>>>>>> REPLACE