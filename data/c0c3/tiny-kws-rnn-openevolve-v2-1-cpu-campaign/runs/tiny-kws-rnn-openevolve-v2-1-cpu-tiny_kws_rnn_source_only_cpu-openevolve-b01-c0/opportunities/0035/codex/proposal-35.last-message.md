MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 118-unit GRU using the passing 23-frame schedule will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the 119-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 119 to 118 while preserving the 23-frame schedule and training procedure.

EVIDENCE: The 119-unit, 23-step model passed at 85.399%, while the 117-unit version narrowly missed at 84.908%; testing the intervening width is the most direct probe of the lower-cost accuracy boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
=======
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
>>>>>>> REPLACE