MECHANISM: Dual-view readout with nested trailing-edge subsampling

HYPOTHESIS: A 111-unit dual-readout GRU omitting frame 29 from the qualified 27-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.7%.

INTENDED_EDIT: Classify from concatenated mean and final recurrent states, and use 26 frames by excluding indices 1, 29, and 30 from the qualified 29-frame base schedule.

EVIDENCE: The 27-step dual-readout design achieved 87.48% accuracy, and removing frame 30 from its 28-step predecessor caused no accuracy loss, supporting a further isolated trailing-edge omission with a 2.48-point margin.

<<<<<<< SEARCH
        self.classifier = nn.Linear(111, 8)
=======
        self.classifier = nn.Linear(222, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 5:
            return [
                frame
                for frame in schedule
                if frame
                not in (1, available_frames - 3, available_frames - 2)
            ]
        return schedule
>>>>>>> REPLACE