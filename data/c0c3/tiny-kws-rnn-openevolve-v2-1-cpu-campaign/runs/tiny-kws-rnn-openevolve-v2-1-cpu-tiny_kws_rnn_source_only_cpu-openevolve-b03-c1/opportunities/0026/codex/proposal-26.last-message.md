MECHANISM: Terminal trimming with online temporal-max readout

HYPOTHESIS: Processing frames 2–28 while augmenting the endpoint-and-mean classifier with an online temporal maximum will retain at least 85% validation accuracy and reduce total inference MACs below the verified 28-step model.

INTENDED_EDIT: Remove one additional terminal frame, track the elementwise maximum recurrent output, and classify from the final state, temporal mean, and temporal maximum.

EVIDENCE: Frames 2–29 achieved 85.28% at 28 steps, while removing leading frame 2 previously caused failure; preserving the early boundary and trimming the terminal boundary is therefore the strongest 27-step candidate, with max pooling adding complementary trajectory evidence at only a small classifier cost.

<<<<<<< SEARCH
        self.classifier = nn.Linear(160, 8)
=======
        self.classifier = nn.Linear(240, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((hidden[:, 0, :], mean_output), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 2))
=======
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((hidden[:, 0, :], mean_output, peak), dim=-1)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))
>>>>>>> REPLACE