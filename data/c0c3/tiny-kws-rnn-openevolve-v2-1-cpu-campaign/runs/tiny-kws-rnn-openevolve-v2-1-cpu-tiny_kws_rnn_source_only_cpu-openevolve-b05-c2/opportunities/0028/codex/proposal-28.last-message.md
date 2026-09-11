MECHANISM: Three-of-four causal frame scheduling with compensatory hidden width

HYPOTHESIS: A 74-unit GRU evaluated on 24 uniformly distributed causal frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 19% versus the current 71-unit, 32-step design.

INTENDED_EDIT: Increase the GRU and triple-readout width from 71 to 74 units, resize the classifier from 213 to 222 inputs, and skip one frame in every four while retaining both temporal endpoints.

EVIDENCE: The full-resolution 74-unit design achieved 87.85% accuracy, the strongest observed result and 2.85 points above threshold; its margin motivates exchanging modest width for a 25% recurrent-step reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 71, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(213, 8)
=======
        self.gru = nn.GRU(20, 74, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 71, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 71, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 71, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 74, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 74, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 74, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [
            index
            for index in range(available_frames)
            if index % 4 != 2
        ]
>>>>>>> REPLACE