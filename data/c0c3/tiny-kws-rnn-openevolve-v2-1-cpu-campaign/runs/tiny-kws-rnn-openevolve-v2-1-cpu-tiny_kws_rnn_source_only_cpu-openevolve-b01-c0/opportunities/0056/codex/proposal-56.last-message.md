MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 99-unit, 23-step GRU will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 100-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 100 to 99 while preserving the verified frame schedule and training procedure.

EVIDENCE: The 100-unit model achieved 86.871% accuracy, and every adjacent reduction from 117 through 100 units passed; its 1.871-point margin makes a one-channel trim the most informative lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE