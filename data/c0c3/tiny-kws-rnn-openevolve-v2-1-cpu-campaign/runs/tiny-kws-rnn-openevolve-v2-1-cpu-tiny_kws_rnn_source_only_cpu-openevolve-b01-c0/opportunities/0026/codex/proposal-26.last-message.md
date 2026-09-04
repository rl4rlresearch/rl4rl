MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 118-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.55% versus the passing 119-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 119 to 118 while preserving the verified schedule and training procedure.

EVIDENCE: Every tested additive-readout width from 128 through 119 has passed, and the latest 119-unit model achieved 86.135% accuracy, leaving a 1.135-point margin for the next isolated one-channel reduction.

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