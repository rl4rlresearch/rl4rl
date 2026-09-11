MECHANISM: Orthonormal gauge removal of the final pre-LayerNorm bias

HYPOTHESIS: Combining the qualified query-only QKV bias with a seven-dimensional mean-zero parameterization of `fc2.bias` will reduce the model to 1627 parameters while retaining at least 99% accuracy, because the omitted all-ones bias direction is exactly canceled by the immediately following final LayerNorm.

INTENDED_EDIT: Apply the proven key/value-bias removal and represent the eight-coordinate MLP output bias using seven learned coefficients in a fixed orthonormal basis perpendicular to the all-ones vector, while preserving the baseline initialization RNG stream.

EVIDENCE: The query-only QKV design achieved 99.92% accuracy at 1628 parameters. Unlike the failed positional gauge edit, this removes a zero-initialized null direction immediately before `ln_f`, requiring no transformation of randomly initialized embeddings.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Preserve the baseline constructor RNG stream while learning only the
        # query bias. Key bias is softmax-invariant and value bias is absorbed
        # by the output-projection bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        bias = torch.cat((self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model)))
        qkv = F.linear(x, self.qkv.weight, bias)
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
        # Construct the baseline layer first to preserve all subsequent RNG
        # state, then remove the bias direction erased by the final LayerNorm.
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))

        identity = torch.eye(d_model)
        unit_mean = torch.full((d_model,), 1.0 / math.sqrt(d_model))
        reflector = identity[:, -1] - unit_mean
        reflector = reflector / reflector.norm()
        basis = identity - 2.0 * torch.outer(reflector, reflector)
        self.register_buffer("fc2_bias_basis", basis[:, :-1], persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.fc2_bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
>>>>>>> REPLACE