MECHANISM: Four-and-a-half-times-width ReLU² feed-forward capacity scaling

HYPOTHESIS: A 4.5× MLP at the proven 512-token SSSL attention geometry will retain at least 490M training tokens while extending the monotonic quality gains from 3× through 4×, reducing val_bpb below 0.993287.

INTENDED_EDIT: Restore quarter-context short attention and widen every MLP from 4× to 4.5× model dimension while preserving the best 50% linear warmdown and 524K-token optimizer batch.

EVIDENCE: Increasing MLP width from 3× to 3.5× improved val_bpb from 0.996996 to 0.995582, and 4× improved it further to 0.993287; this monotonic trend motivates testing the next tensor-core-friendly width above 4×.

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
        mlp_dim = 9 * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context
>>>>>>> REPLACE