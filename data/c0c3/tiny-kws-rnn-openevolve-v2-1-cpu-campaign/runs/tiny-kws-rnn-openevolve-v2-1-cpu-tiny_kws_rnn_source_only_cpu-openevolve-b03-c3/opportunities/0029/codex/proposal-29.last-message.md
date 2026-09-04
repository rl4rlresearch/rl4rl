MECHANISM: Balanced one-unit recurrent-width reduction

HYPOTHESIS: A 45/45/46-unit eight-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 711,071,200 to approximately 702,334,400.

INTENDED_EDIT: Reduce one 46-unit GRU branch to 45 units and resize the aggregate temporal state and classifier from 137 to 136 features.

EVIDENCE: The qualified 45/46/46 model achieved 86.26% accuracy, a 1.26-point margin, and every preceding one-unit branch reduction from 46/47/47 through 45/46/46 remained qualified; this is the smallest further structural cost probe.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 137, 8)
=======
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 136, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 137, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 136, device=device, dtype=dtype)
>>>>>>> REPLACE