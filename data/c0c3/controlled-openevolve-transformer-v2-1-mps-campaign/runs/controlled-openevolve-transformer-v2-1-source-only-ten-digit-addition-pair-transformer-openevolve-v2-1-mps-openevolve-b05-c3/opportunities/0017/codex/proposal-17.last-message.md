MECHANISM: Single-coordinate LayerNorm-to-MLP bias absorption

HYPOTHESIS: Starting from the verified 1601-parameter four-query-bias design, fixing one `ln2` bias coordinate at zero will produce 1600 parameters and retain at least 99% accuracy because `fc1.bias` can represent that coordinate’s downstream affine contribution.

INTENDED_EDIT: Adopt the qualified four-coordinate query bias, then learn seven `ln2` bias coordinates and reconstruct the eighth as zero without changing initialization RNG streams or decoding.

EVIDENCE: Four query-bias coordinates achieved 99.96% at 1601 parameters. Removing all eight `ln2` biases failed at 17.27%, motivating a localized one-coordinate test that retains seven coordinates and the full MLP capacity.

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Five query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
=======
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Four query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 3))
        )
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # Its omitted coordinate is affine-redundant with fc1.bias. Clone the
        # initialized full bias so later construction and RNG behavior are unchanged.
        self.ln2.bias = nn.Parameter(self.ln2.bias[:-1].detach().clone())
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        ln2_bias = torch.cat(
            (self.ln2.bias, self.ln2.bias.new_zeros(1))
        )
        ln2_x = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
        x = x + self.mlp(ln2_x)
        return x
>>>>>>> REPLACE