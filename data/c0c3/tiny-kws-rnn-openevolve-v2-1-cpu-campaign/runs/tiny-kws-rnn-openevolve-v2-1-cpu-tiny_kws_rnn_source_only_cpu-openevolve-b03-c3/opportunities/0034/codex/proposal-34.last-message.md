MECHANISM: Progressive trailing-boundary frame pruning

HYPOTHESIS: The qualified 45/45/46 GRU can omit frame 30 after already omitting frames 0 and 31, retaining at least 85% validation accuracy while reducing execution to 29 recurrent steps and approximately 637,155,590 total MACs.

INTENDED_EDIT: Restore the high-margin 45/45/46 eight-bin backbone and process only frames 1–29 for 32-frame inputs.

EVIDENCE: The 30-step 45/45/46 model achieved 86.87% accuracy at 658,881,860 MACs, and removing the final frame slightly improved accuracy over the 31-step result; this motivates pruning the adjacent trailing frame while preserving 1.87 points of observed margin.

<<<<<<< SEARCH
        self.gru_c = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 135, 8)
=======
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 136, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_c = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 135, device=device, dtype=dtype)
=======
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 136, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(1, available_frames - 2))
>>>>>>> REPLACE