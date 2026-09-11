MECHANISM: Orthogonally stabilized ungated recurrent ensemble

HYPOTHESIS: Three 48-unit tanh RNN streams with normalized temporal readouts will retain at least 85% accuracy while reducing recurrent MACs by approximately 33% versus the 32/32/31 GRU ensemble.

INTENDED_EDIT: Challenge the assumption that three learned gates are necessary at every recurrent update by replacing each GRU with a wider, orthogonally initialized single-transform RNN and per-stream output normalization, while preserving the successful endpoint/mean/maximum readout and 27-step schedule.

EVIDENCE: Parallel GRUs succeeded with both two 48-unit heads (86.13%) and three 36-unit heads (86.01%), while four 29-unit heads failed, suggesting adequate per-stream width matters more than dense coupling. Using 48-unit streams preserves that demonstrated width but removes two of the three recurrent matrix transforms per head.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with asymmetric widths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 31, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(285, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_0 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 31, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 95, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_0, hidden_1, hidden_2, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for gru, hidden in zip(self.gru_heads, hidden_states):
            head_output, head_hidden = gru(
                normalized,
                hidden.unsqueeze(0).contiguous(),
            )
            outputs.append(head_output[:, 0, :])
            next_hidden.append(head_hidden[0])
        output = torch.cat(outputs, dim=-1)
        return (
            *next_hidden,
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        normalized = self.input_norm(frames)
        head_outputs = []
        next_hidden = []
        for gru, hidden in zip(self.gru_heads, hidden_states):
            output, head_hidden = gru(
                normalized,
                hidden.unsqueeze(0).contiguous(),
            )
            head_outputs.append(output)
            next_hidden.append(head_hidden[0])
        outputs = torch.cat(head_outputs, dim=-1)
        return (
            *next_hidden,
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = torch.cat((hidden_0, hidden_1, hidden_2), dim=-1)
        return self.classifier(torch.cat((endpoint, mean_output, peak), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))


def build_model() -> nn.Module:
    return KeywordGRU()
=======
class KeywordRNN(nn.Module):
    """Three wider, orthogonally stabilized causal tanh RNN streams."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.rnn_heads = nn.ModuleList(
            [
                nn.RNN(
                    20,
                    48,
                    num_layers=1,
                    nonlinearity="tanh",
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )
        self.output_norms = nn.ModuleList([nn.LayerNorm(48) for _ in range(3)])
        for rnn in self.rnn_heads:
            nn.init.xavier_uniform_(rnn.weight_ih_l0)
            nn.init.orthogonal_(rnn.weight_hh_l0)
            nn.init.zeros_(rnn.bias_ih_l0)
            nn.init.zeros_(rnn.bias_hh_l0)
        self.classifier = nn.Linear(432, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_0 = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 144, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 144, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_0, hidden_1, hidden_2, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for rnn, output_norm, hidden in zip(
            self.rnn_heads, self.output_norms, hidden_states
        ):
            head_output, head_hidden = rnn(
                normalized,
                hidden.unsqueeze(0).contiguous(),
            )
            outputs.append(output_norm(head_output[:, 0, :]))
            next_hidden.append(head_hidden[0])
        output = torch.cat(outputs, dim=-1)
        return (
            *next_hidden,
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        normalized = self.input_norm(frames)
        head_outputs = []
        next_hidden = []
        for rnn, output_norm, hidden in zip(
            self.rnn_heads, self.output_norms, hidden_states
        ):
            output, head_hidden = rnn(
                normalized,
                hidden.unsqueeze(0).contiguous(),
            )
            head_outputs.append(output_norm(output))
            next_hidden.append(head_hidden[0])
        outputs = torch.cat(head_outputs, dim=-1)
        return (
            *next_hidden,
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        hidden_states = (hidden_0, hidden_1, hidden_2)
        endpoint = torch.cat(
            [
                output_norm(hidden)
                for output_norm, hidden in zip(
                    self.output_norms, hidden_states
                )
            ],
            dim=-1,
        )
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((endpoint, mean_output, peak), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))


def build_model() -> nn.Module:
    return KeywordRNN()
>>>>>>> REPLACE