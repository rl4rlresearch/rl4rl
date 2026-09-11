MECHANISM: Quarter-batch update densification

HYPOTHESIS: A 131,072-token global batch will preserve at least 420M-token throughput while approximately doubling the proven 1,772 optimizer updates, reducing val_bpb below 0.985506.

INTENDED_EDIT: Restore the proven uniform 4.375× MLP and halve both global and device batch sizes so each optimizer step processes 131,072 tokens without gradient accumulation.

EVIDENCE: Halving the global batch from 524,288 to 262,144 tokens increased updates from roughly 900 to 1,772 and improved val_bpb from 0.994296 to 0.985506 despite slightly lower token exposure, while the larger 786,432-token batch performed substantially worse.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        expansion_quarters = 17 if layer_idx < config.n_layer // 2 else 18
        hidden_dim = expansion_quarters * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_dim = 35 * config.n_embd // 8
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 64   # per-device batch size
>>>>>>> REPLACE