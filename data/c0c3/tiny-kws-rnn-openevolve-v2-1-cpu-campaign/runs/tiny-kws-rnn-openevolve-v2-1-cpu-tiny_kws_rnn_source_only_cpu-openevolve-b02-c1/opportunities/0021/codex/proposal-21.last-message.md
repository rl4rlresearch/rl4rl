MECHANISM: Reallocate one recurrent step’s MAC budget into balanced hidden width

HYPOTHESIS: Balanced 61+60 GRUs processing frames 4–31 will retain at least 85% validation accuracy while reducing exact total inference MACs from 671,608,085 to approximately 667,657,780.

INTENDED_EDIT: Remove one additional leading frame, widen the recurrent blocks from 60+59 to 61+60 units, and resize the state, summary, and classifier to 121 features.

EVIDENCE: The 60+59 model remained feasible through three consecutive frame trims, reaching 85.03% at 29 steps, while 59+59 failed; this suggests recurrent width is at its capacity boundary. At 28 steps, 61+60 is the widest near-balanced two-block configuration that still costs less than the current design.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
=======
        self.gru_a = nn.GRU(20, 61, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(121, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 61, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(3, available_frames))
=======
        return list(range(4, available_frames))
>>>>>>> REPLACE