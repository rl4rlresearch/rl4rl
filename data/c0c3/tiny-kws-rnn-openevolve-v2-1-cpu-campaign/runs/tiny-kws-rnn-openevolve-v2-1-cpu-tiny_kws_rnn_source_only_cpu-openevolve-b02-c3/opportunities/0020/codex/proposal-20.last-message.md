MECHANISM: Parallel grouped-GRU temporal subspaces

HYPOTHESIS: Two independent 52-unit GRU branches over the qualified 30-frame schedule will achieve at least 85% validation accuracy while reducing predicted total inference MACs from 727,338,600 to approximately 550,600,960.

INTENDED_EDIT: Replace the single densely connected 94-unit GRU with two parallel 52-unit GRUs, concatenate their temporal outputs, and retain the mean-plus-final readout.

EVIDENCE: Dense 90-, 92-, 94-, and 96-unit GRUs all qualified, suggesting total recurrent width is useful; the failed low-rank experiment challenged dense recurrent mixing but timed out. Standard parallel GRUs cleanly test whether full cross-channel recurrent connectivity is unnecessary while preserving gated dynamics and efficient sequence execution.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 30, 0)
        return list(range(start, available_frames))
=======
class KeywordGRU(nn.Module):
    """Parallel gated recurrent subspaces with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.branches = nn.ModuleList(
            [
                nn.GRU(20, 52, num_layers=1, batch_first=True),
                nn.GRU(20, 52, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(208, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 2, 52, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for index, branch in enumerate(self.branches):
            output, branch_hidden = branch(
                normalized,
                hidden[:, index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(output[:, 0, :])
            next_hidden.append(branch_hidden[0])
        combined = torch.cat(outputs, dim=1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + combined,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        normalized = self.input_norm(frames)
        outputs = []
        next_hidden = []
        for index, branch in enumerate(self.branches):
            branch_outputs, branch_hidden = branch(
                normalized,
                hidden[:, index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(branch_outputs)
            next_hidden.append(branch_hidden[0])
        combined = torch.cat(outputs, dim=2)
        return (
            torch.stack(next_hidden, dim=1),
            summary + combined.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden.reshape(hidden.shape[0], 104)
        return self.classifier(torch.cat((mean_output, final_output), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 30, 0)
        return list(range(start, available_frames))
>>>>>>> REPLACE