MECHANISM: Reduced MLP expansion for compute reallocation

HYPOTHESIS: Reducing each MLP from 4× to 3× width will preserve the proven eight-layer SSSL attention stack while increasing token throughput enough to achieve val_bpb below 0.995558.

INTENDED_EDIT: Change the squared-ReLU MLP hidden width from 4× to 3× model dimension.

EVIDENCE: Attention sparsification produced essentially no throughput gain, GQA reduced parameters but slowed training, and the seven-layer test could not be verified; targeting the dominant MLP matrix compute is the clearest untested way to trade modest capacity for more training tokens without altering attention or optimizer behavior.

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
        self.c_fc = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(3 * config.n_embd, config.n_embd, bias=False)
>>>>>>> REPLACE