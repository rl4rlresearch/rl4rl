MECHANISM: Asymmetric single-head width reduction

HYPOTHESIS: Two 33-unit GRU heads plus one 32-unit head will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.6% versus the verified three-by-33 model.

INTENDED_EDIT: Reduce only the third GRU head to 32 units and resize the separate hidden states, trajectory summaries, and classifier input from 99/297 to 98/294 features.

EVIDENCE: Three 33-unit heads achieved 85.15% accuracy; reducing all three heads by one unit still passed in the preceding 34-to-33 trial, so removing a single unit from only one head is the smallest available structural capacity reduction.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with complementary dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 33, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(297, 8)
=======
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with asymmetric widths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(294, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 3, 33, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 99, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_0 = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_0, hidden_1, hidden_2, summary, peak, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for head_index, gru in enumerate(self.gru_heads):
            head_output, head_hidden = gru(
                normalized,
                hidden[:, head_index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(head_output[:, 0, :])
            next_hidden.append(head_hidden[0])
        output = torch.cat(outputs, dim=-1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        normalized = self.input_norm(frames)
        head_outputs = []
        next_hidden = []
        for head_index, gru in enumerate(self.gru_heads):
            output, head_hidden = gru(
                normalized,
                hidden[:, head_index, :].unsqueeze(0).contiguous(),
            )
            head_outputs.append(output)
            next_hidden.append(head_hidden[0])
        outputs = torch.cat(head_outputs, dim=-1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((hidden.flatten(start_dim=1), mean_output, peak), dim=-1)
        )
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        hidden_0, hidden_1, hidden_2, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = torch.cat((hidden_0, hidden_1, hidden_2), dim=-1)
        return self.classifier(torch.cat((endpoint, mean_output, peak), dim=-1))
>>>>>>> REPLACE