MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 107-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 108-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 108 to 107 while preserving the verified schedule and training procedure.

EVIDENCE: The 108-unit model achieved 87.853% accuracy, and every adjacent batch-64 width reduction from 117 through 108 units passed, making the next one-channel trim the clearest lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
=======
        self.gru = nn.GRU(20, 107, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(107, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 107, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 107, device=device, dtype=dtype)
>>>>>>> REPLACE