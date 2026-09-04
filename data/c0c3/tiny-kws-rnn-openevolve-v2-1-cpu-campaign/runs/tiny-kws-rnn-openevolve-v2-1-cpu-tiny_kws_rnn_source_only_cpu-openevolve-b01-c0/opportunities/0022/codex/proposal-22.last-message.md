MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 122-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 123-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 123 to 122 while preserving the verified frame schedule and training procedure.

EVIDENCE: Every tested additive-readout width from 128 through 123 has passed; the 123-unit model achieved 85.276% accuracy, so another isolated one-channel reduction is the most direct test of the remaining width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(123, 8)
=======
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(122, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
>>>>>>> REPLACE