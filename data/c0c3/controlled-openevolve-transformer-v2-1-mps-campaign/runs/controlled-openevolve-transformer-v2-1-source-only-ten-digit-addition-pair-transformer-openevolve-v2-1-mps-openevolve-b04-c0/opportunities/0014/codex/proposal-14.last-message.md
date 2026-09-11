MECHANISM: Shared token/position embedding gauge anchor

HYPOTHESIS: Sharing one token-embedding coordinate with the corresponding positional-embedding coordinate will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because a common shift between all token embeddings and all positional embeddings leaves transformer inputs unchanged and changes every output logit equally.

INTENDED_EDIT: Replace the two embedding tables with jointly initialized tables whose first scalar is shared, using a gauge-equivalent transformation of the original initialization and the shared token table for output logits.

EVIDENCE: Sharing redundant attention bias pathways retained 99.76% at 1628 parameters, while fixing a positional gauge coordinate reached only 91.63%; this tests an exact embedding gauge through pathway-preserving sharing rather than deleting a coordinate.

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


class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing one gauge-redundant scalar."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(1))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 1))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 1))

        # Match the random-number consumption of two nn.Embedding constructors.
        nn.init.normal_(torch.empty(vocab_size, d_model))
        nn.init.normal_(torch.empty(max_seq_len, d_model))

    def token_weight(self) -> torch.Tensor:
        return torch.cat((self.anchor, self.token_rest)).view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        return torch.cat((self.anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)

    def token(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.token_weight())

    def position(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.pos_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.embeddings = SharedAnchorEmbeddings(cfg.vocab_size, cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
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
        if isinstance(module, SharedAnchorEmbeddings):
            token = module.token_rest.new_empty(module.vocab_size, module.d_model)
            pos = module.pos_rest.new_empty(module.max_seq_len, module.d_model)
            nn.init.normal_(token, mean=0.0, std=0.02)
            nn.init.normal_(pos, mean=0.0, std=0.02)

            # Gauge-transform the ordinary initialization so the two anchors
            # agree while token-plus-position inputs remain unchanged.
            shift = 0.5 * (pos[0, 0] - token[0, 0])
            token[:, 0].add_(shift)
            pos[:, 0].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, 0].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[1:])
                module.pos_rest.copy_(pos.flatten()[1:])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.embeddings.token(idx) + self.embeddings.position(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = F.linear(x, self.embeddings.token_weight())
>>>>>>> REPLACE