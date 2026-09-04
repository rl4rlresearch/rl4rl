MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 113-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 114-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 114 to 113 while preserving the successful schedule and training procedure.

EVIDENCE: Successive batch-64 width reductions from 117 through 114 units all passed, and the 114-unit model achieved 86.871% accuracy, leaving a 1.87-point margin that motivates testing the adjacent lower-cost width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 114, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
=======
        self.gru = nn.GRU(20, 113, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(113, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 114, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 114, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 113, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 113, device=device, dtype=dtype)
>>>>>>> REPLACE