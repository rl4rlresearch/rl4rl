MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 111-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 112-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 112 to 111 while preserving the verified 23-frame schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 112 units passed, and the 112-unit model achieved 85.890% accuracy, leaving a 0.890-point margin that motivates testing the next lower-cost width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
>>>>>>> REPLACE