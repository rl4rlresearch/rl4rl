MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 121-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 122-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 122 to 121 while preserving the verified frame schedule and training procedure.

EVIDENCE: Every tested additive-readout width from 128 through 122 has passed; the 122-unit model achieved 85.644% accuracy, so another isolated one-channel reduction is the most informative test of the width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(122, 8)
=======
        self.gru = nn.GRU(20, 121, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(121, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 121, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
>>>>>>> REPLACE