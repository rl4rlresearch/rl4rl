MECHANISM: Joint one-unit width refinement and single-edge-frame omission

HYPOTHESIS: A 97-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the qualified 98-unit, 31-step design.

INTENDED_EDIT: Reduce the GRU and dual-view pooling width from 98 to 97 and omit the earliest input frame, yielding 31 recurrent steps.

EVIDENCE: The 97-unit dual-view model qualified at 86.01% using all frames, while the 98-unit 31-step model qualified at 86.50%; combining these individually successful, minimal reductions is the closest unresolved cost refinement.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))
>>>>>>> REPLACE