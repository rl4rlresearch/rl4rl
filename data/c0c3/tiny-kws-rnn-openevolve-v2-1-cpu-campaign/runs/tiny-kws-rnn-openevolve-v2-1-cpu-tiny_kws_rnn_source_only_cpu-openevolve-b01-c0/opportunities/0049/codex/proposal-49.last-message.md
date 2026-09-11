MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 106-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 107-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 107 to 106 while preserving the verified schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 107 units passed; the 107-unit model achieved 85.767% accuracy, motivating the next isolated lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 107, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(107, 8)
=======
        self.gru = nn.GRU(20, 106, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(106, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 107, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 107, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 106, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 106, device=device, dtype=dtype)
>>>>>>> REPLACE