MECHANISM: Tensor-Core-aligned MLP capacity expansion

HYPOTHESIS: Restoring five-step Muon refinement and widening each MLP from 4× to 4.25× will retain near-497M-token throughput while reducing val_bpb below 0.995558.

INTENDED_EDIT: Restore the proven five Muon polar iterations and increase the 512-wide model’s MLP hidden dimension from 2048 to 2176 channels.

EVIDENCE: Reducing MLP width to 3.75× saved 1.0M parameters and added only 1.6M tokens while worsening val_bpb from 0.995558 to 0.996902, indicating that MLP capacity is more valuable than this marginal throughput; five Muon iterations also outperformed three at identical reported throughput.

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
        hidden_dim = 17 * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
                momentum=0.95, ns_steps=3, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE