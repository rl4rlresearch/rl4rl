MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 124-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.48% versus the passing 125-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 125 to 124 while preserving the verified schedule and training procedure.

EVIDENCE: Successive additive-readout widths of 127, 126, and 125 units all passed; the 125-unit model achieved 85.031% accuracy, making a one-channel trim the smallest isolated structural test despite its narrow margin.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
=======
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
>>>>>>> REPLACE