MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 104-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 105-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 105 to 104 while preserving the verified schedule and training procedure.

EVIDENCE: Every adjacent batch-64 width reduction from 117 through 105 units passed, and the 105-unit model achieved 86.748% accuracy, leaving a 1.748-point margin that motivates testing the next lower-cost width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 105, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(105, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 105, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE