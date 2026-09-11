MECHANISM: Seven-halves-width ReLU² feed-forward compute rebalancing

HYPOTHESIS: A 3.5× MLP with the best verified 50% linear warmdown will retain more capacity than the 3× design while exceeding the 4× design’s 513.3M-token throughput, reducing val_bpb below 0.993287.

INTENDED_EDIT: Narrow every MLP from 4× to 3.5× model width and restore WARMDOWN_RATIO from 0.4 to the best verified value of 0.5.

EVIDENCE: The 4× design achieved 0.993287 at 513.3M tokens, while 3× increased throughput to 556.3M tokens but regressed modestly to 0.996996; 3.5× directly tests the capacity-throughput midpoint under the strongest verified schedule.

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
        hidden_dim = 7 * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
>>>>>>> REPLACE