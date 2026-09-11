MECHANISM: Batched null-direction parameter reuse with fused bias reduction

HYPOTHESIS: Reusing attention-output column 1’s LayerNorm-invisible mean as final-LayerNorm scale coordinate 2 will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy; batching both shared-scale reductions and eliminating a duplicate projection-bias reduction will allow the previously timed-out design to finish verification.

INTENDED_EDIT: Remove the second dedicated final-LayerNorm scale, derive the two learned scales from attention projection columns 0 and 1, initialize both column means to one, and reuse one computed projection-bias mean within each block.

EVIDENCE: The first null-direction reuse achieved 100% accuracy at 1,268 parameters, so the same independent symmetry remains the strongest supported reduction. The analogous 1,267 design produced no accuracy result because training timed out, motivating an equivalent implementation with fewer redundant reductions.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one trainable scale stored in an attention null direction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(
        self, x: torch.Tensor, shared_scale: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:2],
                self.weight.new_ones(2),
                shared_scale.reshape(1),
                self.weight.new_ones(2),
            )
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with two trainable scales stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(
        self, x: torch.Tensor, shared_scales: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        qk, v = qkv.chunk(2, dim=-1)

        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.proj.bias.mean()
=======
    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        qk, v = qkv.chunk(2, dim=-1)

        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + shared_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = torch.stack(
            (self.attn.proj.bias.mean(), self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_bias = self.attn.proj.bias.mean()
        shared_biases = torch.stack(
            (shared_bias, self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x), shared_bias)
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            shared_columns = self.blocks[-1].attn.proj.weight[:, :2]
            shared_columns.add_(
                1.0 - shared_columns.mean(dim=0, keepdim=True)
            )
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
=======
        shared_final_scales = self.blocks[-1].attn.proj.weight[:, :2].mean(dim=0)
        x = self.ln_f(x, shared_final_scales)
>>>>>>> REPLACE