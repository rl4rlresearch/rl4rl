MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 100-unit, 23-step GRU will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 101-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 101 to 100 while preserving the verified schedule and training procedure.

EVIDENCE: The 101-unit model achieved 85.276% accuracy, and every adjacent reduction from 117 through 101 units passed; the next one-channel trim is the most informative lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 101, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(101, 8)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE