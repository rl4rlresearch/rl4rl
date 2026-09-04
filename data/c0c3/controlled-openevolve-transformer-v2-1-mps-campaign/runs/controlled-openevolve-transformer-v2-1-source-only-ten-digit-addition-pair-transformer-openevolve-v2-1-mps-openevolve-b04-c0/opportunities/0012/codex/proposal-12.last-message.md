MECHANISM: Single-coordinate pre-MLP bias sharing

HYPOTHESIS: Sharing one `ln2` bias coordinate with one `fc1` bias coordinate will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because both biases affect only the same pre-GELU activations, the remaining seven LayerNorm bias coordinates stay independent, and both gradient pathways remain active.

INTENDED_EDIT: Replace `ln2` with a LayerNorm whose final bias coordinate reuses `mlp.fc1.bias[0]`, eliminating one learned scalar without narrowing the model.

EVIDENCE: Sharing all eight `ln2` coordinates with `fc1.bias` scored 15.44%, while deleting all eight scored 14.87%; the successful attention-bias tie shows pathway-preserving sharing can work, so a one-coordinate tie tests whether the earlier failure came from imposing eight coupled constraints at once.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class SharedCoordinateLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor, shared_bias: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, shared_bias.reshape(1)))
        return F.layer_norm(x, (x.size(-1),), self.weight, bias, 1e-5)


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = SharedCoordinateLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x, self.mlp.fc1.bias[0]))
        return x
>>>>>>> REPLACE