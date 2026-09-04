MECHANISM: Parameter-matched SwiGLU content gating

HYPOTHESIS: On the verified 212,992-token, 23/25-rate baseline, replacing unary squared-ReLU features with parameter-matched multiplicative SwiGLU features will retain at least 480M training tokens and reduce val_bpb below 0.983505.

INTENDED_EDIT: Restore the best verified peak learning rates and replace each dense squared-ReLU MLP with a nearly parameter-matched, 64-aligned SwiGLU MLP whose learned gate conditionally selects prediction features.

EVIDENCE: The 23/25-rate design achieved the best val_bpb of 0.983505, while nearby rate refinements, eight-head attention, and cosine warmdown failed to improve it; all retained the load-bearing assumption of an ungated squared-ReLU feature bank, making content-dependent MLP gating an untested computational mechanism.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        else:
            mlp_mult_quarters = 22  # 5.5x
        mlp_dim = mlp_mult_quarters * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x dense-equivalent budget
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x dense-equivalent budget
        else:
            mlp_mult_quarters = 22  # 5.5x dense-equivalent budget
        dense_dim = mlp_mult_quarters * config.n_embd // 4
        # Three SwiGLU matrices match the two dense matrices at 2/3 width.
        mlp_dim = ((2 * dense_dim // 3 + 32) // 64) * 64
        self.c_fc = nn.Linear(config.n_embd, 2 * mlp_dim, bias=False)
        self.c_proj = nn.Linear(mlp_dim, config.n_embd, bias=False)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
EMBEDDING_LR = 0.6 * 13 / 14      # batch-normalized token embedding LR
UNEMBEDDING_LR = 0.004 * 13 / 14  # batch-normalized lm_head LR
MATRIX_LR = 0.04 * 13 / 14        # batch-normalized Muon matrix LR
SCALAR_LR = 0.5 * 13 / 14         # batch-normalized scalar LR
=======
EMBEDDING_LR = 0.6 * 23 / 25      # best verified token embedding peak LR
UNEMBEDDING_LR = 0.004 * 23 / 25  # best verified lm_head peak LR
MATRIX_LR = 0.04 * 23 / 25        # best verified Muon matrix peak LR
SCALAR_LR = 0.5 * 23 / 25         # best verified scalar peak LR
>>>>>>> REPLACE