MECHANISM: More-update training for a 120-unit strided GRU

HYPOTHESIS: A 120-unit GRU trained with batch size 64 on the proven 16-frame even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by about 5.9% versus Reference Design 2.

INTENDED_EDIT: Select every second frame, halve the training batch size, and reduce recurrent, summary, and classifier width from 128 to 120 units.

EVIDENCE: Reference Design 2 reached 85.890% accuracy at width 124 after batch size 64 recovered the deficit seen with fewer optimizer updates; its 0.89-point accuracy margin motivates testing the next four-unit structural reduction while preserving the successful schedule and training regime.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE