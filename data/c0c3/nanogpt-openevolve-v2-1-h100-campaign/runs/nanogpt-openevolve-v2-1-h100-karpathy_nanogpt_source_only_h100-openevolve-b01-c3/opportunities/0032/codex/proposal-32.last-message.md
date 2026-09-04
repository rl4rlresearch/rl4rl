MECHANISM: Tensor-core-aligned MLP capacity expansion

HYPOTHESIS: A 5× MLP at the proven 93.75% learning rates will retain at least 445M-token throughput and reduce val_bpb below 0.984614.

INTENDED_EDIT: Restore the best verified optimizer rates and increase every MLP hidden width from 4.375× to 5×, yielding a 2560-channel hidden dimension aligned to 128-channel tensor-core tiles.

EVIDENCE: The 4.375× MLP achieved 0.994296 versus 0.996902 for 3.75× despite processing 25.7M fewer tokens, showing that additional MLP capacity outweighed throughput; 93.75% learning rates independently produced the best verified baseline at 0.984614.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 35 * config.n_embd // 8
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 5 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
EMBEDDING_LR = 0.4875   # 81.25% of the proven embedding LR
UNEMBEDDING_LR = 0.00325 # 81.25% of the proven lm_head LR
MATRIX_LR = 0.0325      # 81.25% of the proven Muon LR
SCALAR_LR = 0.40625     # 81.25% of the proven per-layer scalar LR
=======
EMBEDDING_LR = 0.5625   # 93.75% of the proven embedding LR
UNEMBEDDING_LR = 0.00375 # 93.75% of the proven lm_head LR
MATRIX_LR = 0.0375      # 93.75% of the proven Muon LR
SCALAR_LR = 0.46875     # 93.75% of the proven per-layer scalar LR
>>>>>>> REPLACE