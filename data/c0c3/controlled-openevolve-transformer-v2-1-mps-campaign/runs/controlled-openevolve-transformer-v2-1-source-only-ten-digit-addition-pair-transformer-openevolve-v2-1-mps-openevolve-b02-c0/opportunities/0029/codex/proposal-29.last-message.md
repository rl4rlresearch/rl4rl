MECHANISM: Incremental dynamic pre-attention LayerNorm bias tying

HYPOTHESIS: Tying the second-to-last `ln1` bias coordinate to the first learned coordinate will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because the following biased QKV projection can absorb independent LayerNorm offsets while the tied coordinate remains dynamically learned.

INTENDED_EDIT: Store six learned `ln1` bias coordinates, reconstruct the seventh from the first, and retain the fixed-zero eighth coordinate without changing the all-zero initialization.

EVIDENCE: The existing one-coordinate `ln1` bias anchor previously achieved 99.95% accuracy, and a learned `ln1` scale tie achieved 99.78%; unlike the failed second scale tie, this edit targets the additive LayerNorm-to-QKV bias redundancy and preserves a trainable signal.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = torch.cat(
            (self.bias, self.bias[:1], self.bias.new_zeros(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE