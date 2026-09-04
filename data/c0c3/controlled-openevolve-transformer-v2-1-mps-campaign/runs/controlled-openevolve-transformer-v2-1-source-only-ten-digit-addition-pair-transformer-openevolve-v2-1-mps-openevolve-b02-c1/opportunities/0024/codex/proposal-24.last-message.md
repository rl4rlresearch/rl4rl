MECHANISM: Optimizer-preserving key-projection LayerNorm quotient

HYPOTHESIS: Reparameterizing all key projections on seven zero-mean LayerNorm coordinates will reduce the model from 1,438 to 1,430 parameters while retaining at least 99% accuracy, because key constants are softmax-invisible and virtual full-width AdamW moments preserve the successful optimization geometry.

INTENDED_EDIT: Replace the full-width key portion of QKV with an eight-by-seven learned projection over non-affine LayerNorm outputs, retain full-width query/value projections and `ln1` affine parameters, and add the omitted key coordinates to quotient-aware optimization.

EVIDENCE: Quotient-aware optimization preserved 99.75% for the MLP residual quotient and 99.93% for the attention-output quotient, whereas removing `ln1` scales failed near 75%; this isolates an exact key-only redundancy without constraining the load-bearing query/value affine pathways.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedQKV(nn.Module):
    """Q/V projections plus keys modulo LayerNorm's all-ones direction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.qv_weight = nn.Parameter(torch.empty(2 * d_model, d_model))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))

        # Preserve the RNG stream of the removed bias-free QKV Linear.
        discarded_weight = torch.empty(3 * d_model, d_model)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    def initialize_from_full_normal(self) -> None:
        full_weight = self.qv_weight.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        with torch.no_grad():
            self.qv_weight[: self.d_model].copy_(
                full_weight[: self.d_model]
            )
            self.qv_weight[self.d_model :].copy_(
                full_weight[2 * self.d_model :]
            )
            self.key_weight.copy_(
                key_weight[:, :-1] - key_weight[:, -1:]
            )

    def forward(
        self, affine_x: torch.Tensor, normalized_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.linear(
            affine_x, self.qv_weight[: self.d_model]
        )
        k = F.linear(normalized_x[..., :-1], self.key_weight)
        v = F.linear(
            affine_x, self.qv_weight[self.d_model :]
        )
        return q, k, v


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = GaugeFixedQKV(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
    def forward(
        self, x: torch.Tensor, normalized_x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q, k, v = self.qkv(x, normalized_x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x,
            self.ln1.normalized_shape,
            weight=None,
            bias=None,
            eps=self.ln1.eps,
        )
        affine_normalized = (
            normalized * self.ln1.weight + self.ln1.bias
        )
        x = x + self.attn(affine_normalized, normalized)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            module.initialize_from_full_normal()
=======
        if isinstance(module, GaugeFixedQKV):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedEmbedding):
            module.initialize_from_full_normal()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
=======
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.attn.qkv.key_weight, 1) for block in model.blocks
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
>>>>>>> REPLACE