MECHANISM: Hierarchical four-frame acoustic recurrence with a slower lexical GRU

HYPOTHESIS: Encoding every frame with a 64-unit RNN reset every four frames, then updating a 64-unit GRU from each ordered block descriptor, will retain at least 85% accuracy while reducing recurrent MACs from 713.2M to approximately 300.4M.

INTENDED_EDIT: Replace the monolithic 86-unit full-rate GRU with a two-timescale recurrent hierarchy: a short-horizon local RNN processes all 32 frames, and a persistent GRU processes eight terminal-plus-mean block summaries.

EVIDENCE: Skipping frames at 16 and 20 steps failed, while 24 and 32 steps passed, indicating that retaining acoustic observations is load-bearing. This patch preserves every frame but challenges the assumption that the entire gated state must update at full rate.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with mean and terminal-state readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(32, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
        ]
=======
class KeywordGRU(nn.Module):
    """A two-timescale causal recurrent keyword model."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.local_rnn = nn.RNN(
            20, 64, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.block_norm = nn.LayerNorm(64)
        self.global_gru = nn.GRU(64, 64, num_layers=1, batch_first=True)
        self.readout_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        global_hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return local_hidden, global_hidden, summary, block_sum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_hidden, global_hidden, summary, block_sum, count = state
        local_output, local_hidden_t = self.local_rnn(
            self.input_norm(frame).unsqueeze(1),
            local_hidden.transpose(0, 1).contiguous(),
        )
        local_output = local_output[:, 0, :]
        local_hidden = local_hidden_t.transpose(0, 1)
        block_sum = block_sum + local_output

        if int(count[0, 0].item()) % 4 == 3:
            block_descriptor = local_output + 0.25 * block_sum
            _, global_hidden_t = self.global_gru(
                self.block_norm(block_descriptor).unsqueeze(1),
                global_hidden.transpose(0, 1).contiguous(),
            )
            global_hidden = global_hidden_t.transpose(0, 1)
            local_hidden = torch.zeros_like(local_hidden)
            block_sum = torch.zeros_like(block_sum)

        return (
            local_hidden,
            global_hidden,
            summary + local_output,
            block_sum,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_hidden, global_hidden, summary, block_sum, count = state
        time_steps = frames.shape[1]
        phase = int(count[0, 0].item()) % 4

        if phase != 0 or time_steps % 4 != 0:
            current_state = state
            for index in range(time_steps):
                current_state = self.recurrent_step(
                    frames[:, index, :], current_state
                )
            return current_state

        batch_size = frames.shape[0]
        blocks = time_steps // 4
        grouped_frames = self.input_norm(frames).reshape(
            batch_size * blocks, 4, 20
        )
        local_outputs, _ = self.local_rnn(grouped_frames)
        local_outputs = local_outputs.reshape(batch_size, blocks, 4, 64)

        block_descriptors = (
            local_outputs[:, :, -1, :] + local_outputs.mean(dim=2)
        )
        _, global_hidden_t = self.global_gru(
            self.block_norm(block_descriptors),
            global_hidden.transpose(0, 1).contiguous(),
        )

        return (
            torch.zeros_like(local_hidden),
            global_hidden_t.transpose(0, 1),
            summary + local_outputs.sum(dim=(1, 2)),
            torch.zeros_like(block_sum),
            count + time_steps,
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        _, global_hidden, summary, _, count = state
        mean_local = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_local, global_hidden[:, 0, :]), dim=-1)
        return self.classifier(self.readout_norm(readout))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(32, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
        ]
>>>>>>> REPLACE