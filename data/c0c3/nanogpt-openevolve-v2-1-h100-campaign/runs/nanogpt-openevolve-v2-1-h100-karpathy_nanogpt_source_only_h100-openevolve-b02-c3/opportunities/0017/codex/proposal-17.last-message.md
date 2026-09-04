MECHANISM: Four-and-a-quarter-width ReLU² feed-forward capacity scaling

HYPOTHESIS: A 4.25× MLP will retain at least 495M training tokens while balancing the 4× model’s superior throughput against the 4.5× model’s added capacity, reducing val_bpb below 0.993287.

INTENDED_EDIT: Widen every MLP from 4× to 4.25× model dimension while preserving the proven attention geometry, optimizer batch, and 50% linear warmdown.

EVIDENCE: The 4× design achieved 0.993287 val_bpb at 513.3M tokens, whereas 4.5× regressed to 0.994453 at 480.2M; 4.25× tests the tensor-core-aligned midpoint within this newly bracketed capacity optimum.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dim = 17 * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)
>>>>>>> REPLACE