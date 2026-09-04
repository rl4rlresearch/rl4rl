MECHANISM: Balanced two-block recurrent width reduction

HYPOTHESIS: Two independent 59-unit GRUs over frames 3–31 will retain at least 85% validation accuracy while reducing total inference MACs from 671,608,085 to approximately 661,745,770.

INTENDED_EDIT: Reduce the 60-unit GRU to 59 units, producing two balanced 59-unit blocks and resizing the recurrent state, temporal summary, and classifier input to 118 features.

EVIDENCE: Reducing the successful 60+60 model to 60+59 improved observed accuracy from 85.28% to 86.50% while lowering cost; removing one unit from the larger remaining block is the smallest recurrent-capacity reduction and is slightly cheaper than an unequal 60+58 split of the same aggregate width.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
=======
        self.gru_a = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
>>>>>>> REPLACE