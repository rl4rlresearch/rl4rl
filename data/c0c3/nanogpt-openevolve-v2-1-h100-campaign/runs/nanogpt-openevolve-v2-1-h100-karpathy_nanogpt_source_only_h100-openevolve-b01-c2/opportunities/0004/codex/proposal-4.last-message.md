MECHANISM: Dense full-context FlashAttention

HYPOTHESIS: Reverting the slower SwiGLU MLP and making all eight layers full-context will beat 0.995558 val_bpb because the controlled SSSS result showed that replacing one full-context layer with half-context attention reduced throughput from 497.0M to 413.7M tokens, while full context also preserves more information.

INTENDED_EDIT: Restore the best-performing ReLU-squared MLP and change the attention pattern from six half-context layers plus two full-context layers to eight full-context layers.

EVIDENCE: Reference Design 1 reached 0.995558 val_bpb and 497.0M tokens with ReLU-squared, while SwiGLU worsened to 1.008305 on 402.7M tokens; Reference Design 3 further showed that adding a half-context layer unexpectedly reduced throughput and worsened val_bpb to 1.008991.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = ((8 * config.n_embd // 3 + 127) // 128) * 128
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        value, gate = self.c_fc(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.c_proj(x)
        return x
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "LLLL" # dense full-context attention in every layer
>>>>>>> REPLACE