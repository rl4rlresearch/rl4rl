MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 101-unit, 23-step GRU will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 102-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 102 to 101 while preserving the verified schedule and training procedure.

EVIDENCE: The 102-unit model achieved 86.748% accuracy, and every adjacent reduction from 117 through 102 units passed; this 1.748-point margin motivates the next isolated lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 102, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(102, 8)
=======
        self.gru = nn.GRU(20, 101, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(101, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
>>>>>>> REPLACE