MECHANISM: Single-position LayerNorm gauge fixing

HYPOTHESIS: Removing one all-ones gauge degree from the first positional embedding will reduce the verified 1,628-parameter design to 1,627 parameters while retaining at least 99% accuracy, because the removed component is erased throughout the pre-LayerNorm residual stack and the minimal reparameterization avoids the disruption observed when all positional rows were compressed.

INTENDED_EDIT: Apply the verified key-bias and `ln2`-bias removals, then represent only the first positional embedding with seven learned coordinates and one fixed coordinate while preserving its initialization distribution modulo the removed shift.

EVIDENCE: The 1,628-parameter design achieved 99.95% accuracy, whereas removing all 23 positional gauge degrees fell to 72.91%; removing only one such degree is the most conservative test of the exact invariance while retaining nearly all original positional optimization geometry.

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


class GaugeFixedPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        full = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full)
        self.first_weight = nn.Parameter(full[0, :-1] - full[0, -1])
        self.rest_weight = nn.Parameter(full[1:].clone())

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_weight, (0, 1)).unsqueeze(0)
        weight = torch.cat((first, self.rest_weight), dim=0)
        return F.embedding(positions, weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            full = module.first_weight.new_empty(module.num_embeddings, module.embedding_dim)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.first_weight.copy_(full[0, :-1] - full[0, -1])
                module.rest_weight.copy_(full[1:])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE