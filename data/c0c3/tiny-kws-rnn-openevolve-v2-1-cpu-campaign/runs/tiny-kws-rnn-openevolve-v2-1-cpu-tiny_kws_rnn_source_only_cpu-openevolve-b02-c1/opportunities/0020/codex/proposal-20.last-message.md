MECHANISM: Wider ungated block-diagonal tanh recurrence

HYPOTHESIS: Two independent 100-unit tanh RNNs over the verified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs below 570 million.

INTENDED_EDIT: Replace the gated 60+59 recurrence with wider 100+100 ungated recurrent blocks, resizing the state, temporal summary, and classifier while preserving causal mean pooling and frames 3–31.

EVIDENCE: Parallel recurrent blocks already achieved 86.50% accuracy before trimming; replacing each GRU’s three gated matrices with one RNN transition permits 200 aggregate hidden features at 24,000 recurrent MACs per step versus 28,383 currently. This challenges the load-bearing assumption that learned gates are necessary for these short normalized sequences.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Asymmetric parallel causal GRUs with a shared online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
=======
class KeywordGRU(nn.Module):
    """Wide block-diagonal tanh recurrence with online temporal averaging."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.rnn_a = nn.RNN(
            20, 100, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.rnn_b = nn.RNN(
            20, 100, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.classifier = nn.Linear(200, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 200, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
=======
        output_a, hidden_a = self.rnn_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.rnn_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
=======
        outputs_a, hidden_a = self.rnn_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.rnn_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE