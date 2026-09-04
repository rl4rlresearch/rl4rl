MECHANISM: Optimizer-preserving query-projection LayerNorm quotient

HYPOTHESIS: Reparameterizing queries on seven zero-mean LayerNorm coordinates will reduce the model from 1,430 to 1,422 parameters while maintaining at least 99% accuracy, because the learned affine transformation can be absorbed into query weights and the independent query bias, while virtual full-width AdamW moments preserve optimization geometry.

INTENDED_EDIT: Split QKV storage into seven-coordinate query and key projections plus a full-width affine value projection, and add the omitted query coordinates to quotient-aware optimization.

EVIDENCE: The analogous key-projection quotient reduced the 1,438-parameter model to 1,430 parameters with 99.95% accuracy. Unlike the load-bearing value pathway with its shared value/output bias, queries already have an independent bias that can absorb the LayerNorm affine offset.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Q/V projections plus keys modulo LayerNorm's all-ones direction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.qv_weight = nn.Parameter(torch.empty(2 * d_model, d_model))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
=======
class GaugeFixedQKV(nn.Module):
    """Q/K projections modulo LayerNorm's all-ones direction plus full-width V."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def initialize_from_full_normal(self) -> None:
        full_weight = self.value_weight.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        query_weight = full_weight[: self.d_model]
        key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        with torch.no_grad():
            self.query_weight.copy_(
                query_weight[:, :-1] - query_weight[:, -1:]
            )
            self.key_weight.copy_(
                key_weight[:, :-1] - key_weight[:, -1:]
            )
            self.value_weight.copy_(
                full_weight[2 * self.d_model :]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = F.linear(
            affine_x, self.qv_weight[: self.d_model]
        )
        k = F.linear(normalized_x[..., :-1], self.key_weight)
        v = F.linear(
            affine_x, self.qv_weight[self.d_model :]
        )
=======
        q = F.linear(normalized_x[..., :-1], self.query_weight)
        k = F.linear(normalized_x[..., :-1], self.key_weight)
        v = F.linear(affine_x, self.value_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        ] + [
            (block.attn.qkv.key_weight, 1) for block in model.blocks
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
=======
        ] + [
            (block.attn.qkv.query_weight, 1) for block in model.blocks
        ] + [
            (block.attn.qkv.key_weight, 1) for block in model.blocks
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
>>>>>>> REPLACE