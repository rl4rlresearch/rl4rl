MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 102-unit, 23-step GRU will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 103-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 103 to 102 while preserving the verified schedule and training procedure.

EVIDENCE: Every adjacent width reduction from 117 through 103 units passed; the 103-unit model achieved 85.276% accuracy, making the next one-channel trim the most informative lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
=======
        self.gru = nn.GRU(20, 102, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(102, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
>>>>>>> REPLACE