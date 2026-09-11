MECHANISM: Bottlenecked stacked tanh recurrence with multi-scale readout

HYPOTHESIS: A 64-unit acoustic RNN feeding a 128-unit temporal RNN will retain at least 85% accuracy while reducing estimated total inference MACs from 478,744,040 to approximately 392,660,480.

INTENDED_EDIT: Replace the single 103-unit GRU with two standard causal tanh RNNs, preserve all 16 scheduled frames, and classify from the mean upper-layer output plus final upper- and lower-layer states.

EVIDENCE: The 101-unit GRU qualified at 85.276%, while reducing coverage to 15 frames collapsed accuracy to 80.25%. This tests whether gated full-width updates—not temporal coverage—are the costly assumption; standard sequence modules also avoid the execution problem encountered by the hierarchical GRU attempt.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
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
        return list(range(0, available_frames, 2))
=======
class KeywordStackedRNN(nn.Module):
    """A bottlenecked two-level tanh recurrence with multi-scale readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.lower_rnn = nn.RNN(
            20, 64, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.bridge_norm = nn.LayerNorm(64)
        self.upper_rnn = nn.RNN(
            64, 128, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.output_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(320, 8)

        for recurrent in (self.lower_rnn, self.upper_rnn):
            nn.init.xavier_uniform_(recurrent.weight_ih_l0)
            nn.init.orthogonal_(recurrent.weight_hh_l0)
            nn.init.zeros_(recurrent.bias_ih_l0)
            nn.init.zeros_(recurrent.bias_hh_l0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_hidden = torch.zeros(
            batch_size, 1, 64, device=device, dtype=dtype
        )
        upper_hidden = torch.zeros(
            batch_size, 1, 128, device=device, dtype=dtype
        )
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return lower_hidden, upper_hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_hidden, upper_hidden, summary, count = state
        lower_output, lower_hidden = self.lower_rnn(
            self.input_norm(frame).unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_output, upper_hidden = self.upper_rnn(
            self.bridge_norm(lower_output),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        upper_output = upper_output[:, 0, :]
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            summary + upper_output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_hidden, upper_hidden, summary, count = state
        lower_outputs, lower_hidden = self.lower_rnn(
            self.input_norm(frames),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_outputs, upper_hidden = self.upper_rnn(
            self.bridge_norm(lower_outputs),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            summary + upper_outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        lower_hidden, upper_hidden, summary, count = state
        features = torch.cat(
            (
                self.output_norm(summary / count.clamp_min(1.0)),
                self.output_norm(upper_hidden[:, 0, :]),
                self.bridge_norm(lower_hidden[:, 0, :]),
            ),
            dim=-1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return KeywordStackedRNN()
>>>>>>> REPLACE