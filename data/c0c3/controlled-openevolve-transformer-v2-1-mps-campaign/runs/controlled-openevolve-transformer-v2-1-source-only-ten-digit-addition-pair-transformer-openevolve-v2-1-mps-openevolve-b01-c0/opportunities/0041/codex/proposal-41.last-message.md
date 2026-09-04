MECHANISM: Coordinate-anchored softmax-null key-row gauge

HYPOTHESIS: Fixing one key-row coordinate to zero will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because LayerNorm confines inputs to an affine hyperplane and the omitted direction changes that key coordinate only by a position-independent offset, which causal softmax removes.

INTENDED_EDIT: Replace the first key-projection row with seven directly represented coordinates and a fixed zero eighth coordinate, reconstructing a functionally equivalent initialization while preserving the original RNG sequence.

EVIDENCE: The centered key-row gauge tested the same exact redundancy but reached only 70.16%; its zero-sum reconstruction couples every learned coordinate to the eighth weight. This coordinate-anchored basis removes that coupling while retaining the verified 1619-parameter design’s successful elimination of all key biases.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class AnchoredKeyProjection(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original fused projection.
        _ = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.key_row = nn.Parameter(torch.empty(d_model - 1))
        self.other_rows = nn.Parameter(torch.empty(3 * d_model - 1, d_model))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        key_row = weight[self.d_model]
        with torch.no_grad():
            self.key_row.copy_(key_row[:-1] - key_row[-1])
            self.other_rows.copy_(
                torch.cat(
                    (weight[: self.d_model], weight[self.d_model + 1 :]),
                    dim=0,
                )
            )

    def reconstructed_weight(self) -> torch.Tensor:
        key_row = torch.cat((self.key_row, self.key_row.new_zeros(1)))
        return torch.cat(
            (
                self.other_rows[: self.d_model],
                key_row.unsqueeze(0),
                self.other_rows[self.d_model :],
            ),
            dim=0,
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Construct with the original bias first to preserve the baseline RNG
        # sequence, then retain only the functionally relevant q/v biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
        self.qkv.bias = None
=======
        # Preserve the original fused-linear constructor RNG while anchoring
        # one softmax-null key-row coordinate.
        self.qkv = AnchoredKeyProjection(d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, fused_bias)
=======
        qkv = F.linear(x, self.qkv.reconstructed_weight(), fused_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, AnchoredKeyProjection):
            full_weight = module.key_row.new_empty(
                3 * module.d_model, module.d_model
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE