MECHANISM: Adjacent leading-frame pruning on the qualified spectral trunk

HYPOTHESIS: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy on frames 4–28, reducing execution from 26 to 25 recurrent steps and total inference MACs from 540,423,240 to approximately 519,679,860.

INTENDED_EDIT: Adopt the qualified 17-band recurrent input and remove the earliest frame from the successful 26-step frames-3–28 schedule, while preserving recurrent width, dual-view classification, and training procedure.

EVIDENCE: Frames 3–28 with bands 1–17 achieved 85.89% accuracy at 540,423,240 MACs; the preceding successful transition from frames 2–28 to frames 3–28 preserved identical accuracy, making another leading-frame removal the closest supported reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 84, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame[..., :19]).unsqueeze(1),
=======
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames[..., :19]), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(1, available_frames - 2))
=======
        return list(range(4, available_frames - 3))
>>>>>>> REPLACE