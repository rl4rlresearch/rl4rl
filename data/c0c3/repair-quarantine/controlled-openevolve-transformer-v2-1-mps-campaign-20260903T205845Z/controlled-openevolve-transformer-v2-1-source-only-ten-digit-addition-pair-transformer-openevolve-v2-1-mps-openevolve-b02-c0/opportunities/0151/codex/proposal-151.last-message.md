MECHANISM: LayerNorm–query-bias parameter sharing

HYPOTHESIS: Sharing one query-bias coordinate with the retained fifth `ln1` bias will reduce the model to 1,575 parameters while preserving at least 99% accuracy, because `ln1` bias already supplies a learned query-offset path and both scalars initialize identically at zero.

INTENDED_EDIT: Remove the final independent QKV bias parameter, reconstruct that query-bias coordinate from `ln1.bias[-1]`, and pass the shared scalar through attention.

EVIDENCE: The 1,576-parameter model achieved 99.92% accuracy, while fixing the fifth `ln1` bias at zero fell to 73.09%; sharing a downstream redundant query bias retains that empirically important LayerNorm coordinate instead of anchoring it.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key, value, and upstream bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-17].clone())

    def forward(
        self, x: torch.Tensor, shared_query_bias: torch.Tensor
    ) -> torch.Tensor:
        query_bias = torch.cat((self.bias, shared_query_bias))
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                query_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
=======
    def forward(
        self, x: torch.Tensor, shared_query_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x, shared_query_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1.bias[-1:])
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE