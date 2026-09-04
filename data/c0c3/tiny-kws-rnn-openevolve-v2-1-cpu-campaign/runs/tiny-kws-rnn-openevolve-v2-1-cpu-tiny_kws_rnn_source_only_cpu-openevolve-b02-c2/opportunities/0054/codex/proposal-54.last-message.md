MECHANISM: Additive dual-view readout compression

HYPOTHESIS: Averaging the mean and final recurrent outputs on the qualified 25-step schedule will retain at least 85% accuracy while reducing total inference MACs from 519,679,860 to approximately 519,132,180 and parameters from 27,342 to 26,670.

INTENDED_EDIT: Use frames 4–28 and replace the 168-feature concatenated readout with an 84-feature average of the same two temporal views.

EVIDENCE: The unchanged 84-unit GRU qualified at 85.03% on frames 4–28; compressing only its untested readout preserves both recurrent statistics while avoiding the spectral and temporal reductions that recently failed.

<<<<<<< SEARCH
        self.classifier = nn.Linear(168, 8)
=======
        self.classifier = nn.Linear(84, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 3))
=======
        pooled = 0.5 * (mean_output + hidden[:, 0, :])
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(4, available_frames - 3))
>>>>>>> REPLACE