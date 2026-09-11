MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 103-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 103 while preserving the verified schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 104 units passed; the 104-unit model achieved 85.521% accuracy, making the next one-channel trim the most direct lower-cost boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
>>>>>>> REPLACE