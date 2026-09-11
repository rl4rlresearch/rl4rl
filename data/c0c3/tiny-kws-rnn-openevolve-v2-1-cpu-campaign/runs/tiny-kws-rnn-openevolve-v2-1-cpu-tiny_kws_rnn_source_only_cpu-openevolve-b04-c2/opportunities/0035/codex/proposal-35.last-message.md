MECHANISM: Incremental recurrent-width reduction

HYPOTHESIS: An 85-unit GRU trained at batch size 16 on frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs from 543,926,110 to approximately 480,629,950.

INTENDED_EDIT: Replace the current 103-unit, 21-step model with an 85-unit GRU processing 22 contiguous frames, and use batch size 16.

EVIDENCE: Reducing the 22-step GRU from 97 to 91 units lowered accuracy by only 0.37 points, from 86.50% to 86.13%, leaving 1.13 points of margin; repeating that six-unit reduction is the most direct test of a substantially cheaper frontier.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 16
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(85, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))
=======
        if available_frames <= 11:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
>>>>>>> REPLACE