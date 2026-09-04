MECHANISM: Dual-view mean-and-terminal recurrent readout

HYPOTHESIS: A learned classifier over both the temporal mean and final hidden state will recover the 0.34-point accuracy shortfall of the tested 27-step frames 4–29,31 schedule, reaching at least 85% while keeping recurrent MACs near 836,344,035.

INTENDED_EDIT: Process frames 4–29 and 31, and classify the concatenated mean recurrent output and terminal GRU state with a 206-to-8 linear layer.

EVIDENCE: The 28-step frames 4–31 model achieved 85.40%, while the otherwise similar 27-step frames 4–29,31 model reached 84.66%; retaining frame 31 but exposing its terminal recurrent representation separately targets that narrow loss.

<<<<<<< SEARCH
        self.classifier = nn.Linear(103, 8)
=======
        self.classifier = nn.Linear(206, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 2)) + [available_frames - 1]
>>>>>>> REPLACE