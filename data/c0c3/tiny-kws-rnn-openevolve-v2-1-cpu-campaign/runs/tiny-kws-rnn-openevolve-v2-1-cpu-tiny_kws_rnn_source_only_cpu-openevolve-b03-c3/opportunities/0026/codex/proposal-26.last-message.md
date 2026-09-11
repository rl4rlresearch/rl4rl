MECHANISM: Balanced one-unit recurrent-width reduction

HYPOTHESIS: A 46/46/47-unit eight-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 737,594,560 to approximately 728,701,280.

INTENDED_EDIT: Reduce the qualified 46/47/47 architecture by one unit in a 47-unit branch, balancing widths at 46/46/47 and resizing the temporal state and classifier to 139 features.

EVIDENCE: The 46/47/47 eight-bin design achieved 86.13% accuracy at 737,594,560 MACs; its 1.13-point margin supports the smallest further capacity probe, and balancing the three branches minimizes quadratic recurrent cost at the new aggregate width.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 142, 8)
=======
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 139, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 142, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 139, device=device, dtype=dtype)
>>>>>>> REPLACE