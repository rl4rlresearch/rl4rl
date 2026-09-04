MECHANISM: Near-full MLP compute reallocation

HYPOTHESIS: Restoring SSSL and reducing MLP expansion from 4× to 3.75× will retain enough capacity to beat 0.995558 val_bpb while processing more than 497M tokens.

INTENDED_EDIT: Restore the proven SSSL attention pattern and use a Tensor-Core-aligned 1920-channel MLP at model width 512.

EVIDENCE: The 3× MLP increased throughput from 497.0M to 528.5M tokens but worsened val_bpb by 0.004079; a gentler 3.75× reduction tests whether a smaller throughput gain can be captured without the larger capacity penalty.

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
        hidden_dim = 15 * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # half context throughout, with the final layer forced to full context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
>>>>>>> REPLACE