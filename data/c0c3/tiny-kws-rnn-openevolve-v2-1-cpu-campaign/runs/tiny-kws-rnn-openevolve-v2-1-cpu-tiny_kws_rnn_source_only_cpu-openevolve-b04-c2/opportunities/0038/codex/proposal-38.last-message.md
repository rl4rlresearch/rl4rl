MECHANISM: Incremental trailing-frame trim at fixed recurrent width

HYPOTHESIS: The qualified 86-unit GRU will retain at least 85% validation accuracy on frames 4–23 while reducing execution from 21 to 20 recurrent steps and total inference MACs to approximately 446 million.

INTENDED_EDIT: Change the current model to an 86-unit GRU and classifier, and shorten the standard schedule to 20 contiguous frames while retaining batch size 16 and the established training procedure.

EVIDENCE: The 86-unit, 21-step design achieved 85.15% accuracy at 468,621,740 MACs; analogously, trimming the 103-unit model from 21 to 20 steps reduced accuracy by only 0.12 percentage points and still qualified.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(91, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(86, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames <= 11:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
=======
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))
>>>>>>> REPLACE