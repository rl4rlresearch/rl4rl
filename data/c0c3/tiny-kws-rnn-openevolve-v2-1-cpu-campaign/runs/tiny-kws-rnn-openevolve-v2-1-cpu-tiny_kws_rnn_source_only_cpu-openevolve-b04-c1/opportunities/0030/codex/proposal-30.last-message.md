MECHANISM: Hierarchical multi-rate fast/slow GRU

HYPOTHESIS: A 42-unit fast GRU processing all 28 scheduled frames plus a 52-unit context GRU updated every fourth fast output will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 15.4% versus the verified 58-unit single-rate GRU.

INTENDED_EDIT: Replace the homogeneous recurrent state with full-rank fast and slow GRUs, maintain separate mean/maximum/final summaries for both timescales, and update the command-scale state only seven times per recording.

EVIDENCE: The 28-step 58-unit GRU reached 86.99% accuracy, leaving a 1.99-point margin. The failed MGU showed that weakening the gated update itself was harmful; this alternative retains standard GRUs while challenging the assumption that every unit must perform an equally expensive update at every frame.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, maximum, and final temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
class KeywordGRU(nn.Module):
    """A causal hierarchy with distinct phonetic and command timescales."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRU(20, 42, num_layers=1, batch_first=True)
        self.slow_gru = nn.GRU(42, 52, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(282, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        fast_hidden = torch.zeros(batch_size, 1, 42, device=device, dtype=dtype)
        slow_hidden = torch.zeros(batch_size, 1, 52, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 42, device=device, dtype=dtype)
        fast_maximum = torch.zeros(batch_size, 42, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 52, device=device, dtype=dtype)
        slow_maximum = torch.zeros(batch_size, 52, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        slow_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return hidden.transpose(0, 1), summary + output, maximum, count + 1.0
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        ) = state

        fast_output, fast_hidden = self.fast_gru(
            self.input_norm(frame).unsqueeze(1),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_output = fast_output[:, 0, :]
        fast_maximum = torch.where(
            count > 0,
            torch.maximum(fast_maximum, fast_output),
            fast_output,
        )
        count = count + 1.0

        if int(count[0, 0].item()) % 4 == 0:
            slow_output, slow_hidden = self.slow_gru(
                fast_output.unsqueeze(1),
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_output = slow_output[:, 0, :]
            slow_maximum = torch.where(
                slow_count > 0,
                torch.maximum(slow_maximum, slow_output),
                slow_output,
            )
            slow_summary = slow_summary + slow_output
            slow_count = slow_count + 1.0

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary + fast_output,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        sequence_maximum = outputs.amax(dim=1)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, sequence_maximum),
            sequence_maximum,
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            maximum,
            count + frames.shape[1],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        ) = state

        fast_outputs, fast_hidden = self.fast_gru(
            self.input_norm(frames),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        sequence_maximum = fast_outputs.amax(dim=1)
        fast_maximum = torch.where(
            count > 0,
            torch.maximum(fast_maximum, sequence_maximum),
            sequence_maximum,
        )

        start_step = int(count[0, 0].item())
        slow_positions = [
            index
            for index in range(frames.shape[1])
            if (start_step + index + 1) % 4 == 0
        ]
        if slow_positions:
            slow_outputs, slow_hidden = self.slow_gru(
                fast_outputs[:, slow_positions, :],
                slow_hidden.transpose(0, 1).contiguous(),
            )
            sequence_slow_maximum = slow_outputs.amax(dim=1)
            slow_maximum = torch.where(
                slow_count > 0,
                torch.maximum(slow_maximum, sequence_slow_maximum),
                sequence_slow_maximum,
            )
            slow_summary = slow_summary + slow_outputs.sum(dim=1)
            slow_count = slow_count + len(slow_positions)

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary + fast_outputs.sum(dim=1),
            fast_maximum,
            slow_summary,
            slow_maximum,
            count + frames.shape[1],
            slow_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            fast_maximum,
            slow_summary,
            slow_maximum,
            count,
            slow_count,
        ) = state
        pooled = torch.cat(
            (
                fast_summary / count.clamp_min(1.0),
                fast_maximum,
                fast_hidden[:, 0, :],
                slow_summary / slow_count.clamp_min(1.0),
                slow_maximum,
                slow_hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(pooled)
>>>>>>> REPLACE