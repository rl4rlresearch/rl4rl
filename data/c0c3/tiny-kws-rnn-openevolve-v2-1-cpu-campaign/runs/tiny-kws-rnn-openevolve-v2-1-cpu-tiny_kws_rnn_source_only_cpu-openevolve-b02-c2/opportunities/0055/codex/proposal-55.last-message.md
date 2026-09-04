MECHANISM: Interior-frame pruning with recurrent-width compensation

HYPOTHESIS: An 85-unit GRU processing frames 4–28 except frame 16 will retain at least 85% accuracy while reducing total inference MACs from 519,679,860 to approximately 509,864,000.

INTENDED_EDIT: Preserve both qualified temporal endpoints and all 17 mel features, omit one central frame, and spend part of the saved recurrent budget on one additional hidden unit.

EVIDENCE: The 84-unit frames-4–28 model qualified at 85.03%, while removing frame 4 or boundary-adjacent frames failed; this motivates pruning a more redundant interior frame. An 85-unit, 24-step model remains about 9.8 million MACs cheaper than the qualified design.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames - 3))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return [
            index
            for index in range(4, available_frames - 3)
            if index != available_frames // 2
        ]
>>>>>>> REPLACE