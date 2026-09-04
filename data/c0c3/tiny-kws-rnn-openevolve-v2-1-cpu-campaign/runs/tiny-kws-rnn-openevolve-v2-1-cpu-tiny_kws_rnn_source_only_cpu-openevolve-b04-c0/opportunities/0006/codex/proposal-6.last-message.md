MECHANISM: Dual temporal-mean and final-state readout

HYPOTHESIS: Concatenating the final GRU state with the temporal mean will recover the 0.34-point accuracy deficit of the 31-frame model and reach at least 85%, while retaining nearly all of its 3.125% recurrent-MAC reduction.

INTENDED_EDIT: Process frames 1–31 and classify from a learned combination of the 104-dimensional temporal mean and final recurrent state.

EVIDENCE: The 31-frame model reached 84.66% with lower cross-entropy than the qualifying 32-frame model, so a richer low-cost readout is a targeted way to recover the small accuracy gap without restoring the omitted recurrent step.

<<<<<<< SEARCH
        self.classifier = nn.Linear(104, 8)
=======
        self.classifier = nn.Linear(208, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE