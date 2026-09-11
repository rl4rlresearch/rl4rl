MECHANISM: Spectral-velocity Elman recurrence with dual temporal readout

HYPOTHESIS: A 192-state tanh RNN processing all 32 frames with explicit adjacent-frame deltas and both mean and terminal-state predictions will retain at least 85% accuracy while reducing estimated dense inference MACs from 1.251B to approximately 1.164B.

INTENDED_EDIT: Replace the gated 128-unit GRU with a wider, cheaper 192-unit vanilla RNN; restore full temporal resolution; track normalized spectral velocity; and add a zero-initialized terminal-state classifier alongside the existing mean-state prediction.

EVIDENCE: The full 32-frame GRU achieved 87.24%, while uniform thinning to 24 frames fell to 83.19% and subsequent width/step reductions reached a narrow accuracy boundary. This challenges the shared assumptions that temporal samples should be removed and that mean-pooled gated recurrence is required: the new cell costs 44,544 recurrent MACs per step versus 56,832 for the GRU while preserving every frame and exposing ordered final-state information directly to the prediction.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
class KeywordRNN(nn.Module):
    """A causal spectral-velocity RNN with mean and terminal readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.rnn = nn.RNN(
            40, 192, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.classifier = nn.Linear(192, 8)
        self.terminal_classifier = nn.Linear(192, 8, bias=False)
        nn.init.zeros_(self.terminal_classifier.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 192, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 192, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, previous, summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frame)
        velocity = 0.5 * (normalized - previous)
        features = torch.cat((normalized, velocity), dim=-1)
        output, hidden = self.rnn(
            features.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            normalized,
            summary + output,
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frames)
        prior = torch.cat(
            (previous.unsqueeze(1), normalized[:, :-1, :]), dim=1
        )
        velocity = 0.5 * (normalized - prior)
        features = torch.cat((normalized, velocity), dim=-1)
        outputs, hidden = self.rnn(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            normalized[:, -1, :],
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, _previous, summary, count = state
        mean_logits = self.classifier(summary / count.clamp_min(1.0))
        terminal_logits = self.terminal_classifier(hidden[:, 0, :])
        return mean_logits + terminal_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return KeywordRNN()
>>>>>>> REPLACE