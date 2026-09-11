MECHANISM: Single-edge-frame omission with capacity insurance

HYPOTHESIS: A 98-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs to approximately 877,771,300 and recurrent steps from 26,080 to 25,265.

INTENDED_EDIT: Increase recurrent width from 97 to 98, retain mean-plus-terminal classification, and omit only the earliest input frame.

EVIDENCE: The 98-unit dual-view model achieved 87.12% accuracy across all 32 frames, providing 2.12 percentage points of margin; because larger reductions to 24 or 28 steps failed, skipping a single likely edge-context frame is the most conservative unresolved temporal-cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE