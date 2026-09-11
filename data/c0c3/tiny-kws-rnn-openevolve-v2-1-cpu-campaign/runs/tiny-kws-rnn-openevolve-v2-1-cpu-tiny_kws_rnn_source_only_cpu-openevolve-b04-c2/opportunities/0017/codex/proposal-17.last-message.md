MECHANISM: Near-iso-MAC width-for-step exchange

HYPOTHESIS: A 105-unit GRU processing frames 4–30 will recover the 103-unit model’s 0.34-point accuracy shortfall at 27 steps, reaching at least 85% accuracy while reducing total inference MACs from 867,991,300 to approximately 867,131,475.

INTENDED_EDIT: Drop the final scheduled frame and reinvest the saved recurrent computation in two additional GRU units, preserving the existing pooling and training procedure.

EVIDENCE: The 103-unit frames 4–30 model narrowly missed at 84.66%, while the 103-unit frames 4–31 model reached 85.40%. Among tested 27-step schedules, frames 4–30 also had the lowest cross-entropy, making a small capacity increase the most targeted next test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 105, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(105, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 105, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 1))
>>>>>>> REPLACE