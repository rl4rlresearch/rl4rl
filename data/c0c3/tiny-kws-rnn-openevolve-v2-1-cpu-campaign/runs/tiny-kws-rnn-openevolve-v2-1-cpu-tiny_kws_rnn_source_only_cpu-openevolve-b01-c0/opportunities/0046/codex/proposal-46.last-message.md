MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 109-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 110-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 110 to 109 while preserving the verified schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 110 units passed, and the 110-unit model achieved 86.994% accuracy, leaving a 1.994-point margin that motivates testing the next lower-cost width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
=======
        self.gru = nn.GRU(20, 109, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(109, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 109, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 109, device=device, dtype=dtype)
>>>>>>> REPLACE