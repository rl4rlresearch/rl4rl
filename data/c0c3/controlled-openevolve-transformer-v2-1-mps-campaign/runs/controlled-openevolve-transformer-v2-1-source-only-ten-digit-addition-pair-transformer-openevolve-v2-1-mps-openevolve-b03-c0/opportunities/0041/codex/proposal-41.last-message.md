MECHANISM: Gauge-aware anchored positional-embedding coordinate

HYPOTHESIS: Removing one globally shift-invariant positional-embedding coordinate and preserving its implicit AdamW state will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps.

INTENDED_EDIT: Store the positional-embedding matrix as all but one anchored coordinate, reconstruct the final coordinate as zero, optimize it with the existing gauge-aware optimizer, and add 1,000 low-rate refinement steps.

EVIDENCE: Gauge-aware anchoring of the attention projection bias retained 99.89% at 1,607 parameters; positional embeddings have an independent exact uniform-shift invariance through the pre-normalized block and final LayerNorm, making this a direct test of the successful optimizer-preserving mechanism without repeating the failed MLP-bias reduction.

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


class GaugePositionEmbedding(nn.Module):
    """Position embeddings with one global uniform-shift coordinate anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(weight, mean=0.0, std=0.02)
        weight.sub_(weight[-1, -1].clone())
        self.weight = nn.Parameter(weight.reshape(-1)[:-1].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_zeros(1)))
        weight = weight.view(self.num_embeddings, self.embedding_dim)
        return F.embedding(idx, weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [block.attn.proj.bias for block in model.blocks]
=======
    gauge_parameters = [
        *[block.attn.proj.bias for block in model.blocks],
        model.pos_emb.weight,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=9000)
=======
    p.add_argument("--train-steps", type=int, default=10000)
>>>>>>> REPLACE