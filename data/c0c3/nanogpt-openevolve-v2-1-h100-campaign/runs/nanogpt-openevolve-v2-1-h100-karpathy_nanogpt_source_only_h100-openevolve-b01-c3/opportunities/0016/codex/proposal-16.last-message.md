MECHANISM: Tensor-Core-aligned MLP capacity scaling

HYPOTHESIS: Expanding the MLP from 4× to 4.5× at the proven 524,288-token batch will reduce val_bpb below 0.994753 while processing at least 450M tokens.

INTENDED_EDIT: Restore the successful 524,288-token batch and widen each 512-dimensional MLP to 2304 hidden channels.

EVIDENCE: Increasing MLP width from 4× to 4.25× improved val_bpb from 0.995558 to 0.994753 despite reducing throughput from 497.0M to 476.1M tokens, indicating that additional MLP capacity was more valuable than the lost token count.

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
        hidden_dim = 9 * config.n_embd // 2
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
>>>>>>> REPLACE