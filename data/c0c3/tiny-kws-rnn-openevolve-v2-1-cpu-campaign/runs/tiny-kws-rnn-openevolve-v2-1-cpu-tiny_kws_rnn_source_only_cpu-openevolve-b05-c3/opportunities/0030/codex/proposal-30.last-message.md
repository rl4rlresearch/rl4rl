MECHANISM: Bottleneck-gated highway recurrence

HYPOTHESIS: A 180-unit recurrent cell with one low-rank state-aware update gate will retain at least 85% accuracy under the proven batch-64 training regime while reducing estimated validation inference MACs from 748,482,960 to approximately 694,771,200.

INTENDED_EDIT: Replace the 120-unit three-gate GRU with a wider 180-unit tanh recurrence whose update gate uses input projection plus an 8-unit recurrent bottleneck; preserve the qualified 20-frame mean/endpoint readout and switch to batch size 64.

EVIDENCE: The 114-unit GRU reached 86.01% at 748,482,960 MACs with batch size 64. This tests the load-bearing assumption that three full recurrent gate matrices are necessary: the alternative spends fewer MACs on gating while providing substantially more recurrent state capacity.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
        self.endpoint_classifier = nn.Linear(120, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
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
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
=======
class KeywordHighwayRNN(nn.Module):
    """A wide recurrent state controlled by one low-rank update gate."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_candidate = nn.Linear(20, 180)
        self.recurrent_candidate = nn.Linear(180, 180, bias=False)
        self.input_gate = nn.Linear(20, 180)
        self.gate_reduce = nn.Linear(180, 8)
        self.gate_expand = nn.Linear(8, 180, bias=False)
        self.classifier = nn.Linear(180, 8)
        self.endpoint_classifier = nn.Linear(180, 8)

        nn.init.orthogonal_(self.recurrent_candidate.weight)
        nn.init.zeros_(self.gate_expand.weight)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 180, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 180, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        normalized = self.input_norm(frame)
        candidate = torch.tanh(
            self.input_candidate(normalized) + self.recurrent_candidate(hidden)
        )
        gate_context = torch.tanh(self.gate_reduce(hidden))
        retain = torch.sigmoid(
            self.input_gate(normalized) + self.gate_expand(gate_context)
        )
        hidden = retain * hidden + (1.0 - retain) * candidate
        return hidden, summary + hidden, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(mean_output) + self.endpoint_classifier(hidden)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return KeywordHighwayRNN()
>>>>>>> REPLACE