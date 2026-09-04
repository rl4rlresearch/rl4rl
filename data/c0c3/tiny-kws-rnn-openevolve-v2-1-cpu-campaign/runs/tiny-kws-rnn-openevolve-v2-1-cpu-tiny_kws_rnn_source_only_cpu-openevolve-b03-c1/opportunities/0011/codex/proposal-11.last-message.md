MECHANISM: Fused single-transform Elman recurrence with endpoint-and-trajectory readout

HYPOTHESIS: A 160-unit orthogonally initialized tanh RNN processing all 32 frames will reach at least 85% validation accuracy while reducing recurrent MACs by 20% versus the 100-unit GRU; jointly classifying the final state and temporal mean will offset the loss of GRU gating.

INTENDED_EDIT: Replace the 100-unit GRU with a fused 160-unit standard RNN, preserve full-rate causal processing, and classify from the concatenated final hidden state and mean recurrent output.

EVIDENCE: Full-rate GRUs consistently cleared 85% while every 16–28-frame schedule failed, so all frames should be retained. The prior 160-unit shared-transform recurrence targeted lower MACs but timed out; using the evaluator-supported fused nn.RNN sequence path tests the same efficient single-transform idea without its training-speed failure.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))


def build_model() -> nn.Module:
    return KeywordGRU()
=======
class KeywordRNN(nn.Module):
    """A full-rate causal RNN with endpoint-and-trajectory readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.rnn = nn.RNN(
            20,
            160,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        self.classifier = nn.Linear(320, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 160, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 160, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.rnn(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.rnn(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((hidden[:, 0, :], mean_output), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))


def build_model() -> nn.Module:
    return KeywordRNN()
>>>>>>> REPLACE