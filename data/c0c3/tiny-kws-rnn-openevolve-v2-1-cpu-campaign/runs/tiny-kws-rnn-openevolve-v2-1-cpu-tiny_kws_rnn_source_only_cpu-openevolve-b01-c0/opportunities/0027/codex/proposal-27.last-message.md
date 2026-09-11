MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 117-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.56% versus the passing 118-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 118 to 117 while preserving the verified frame schedule and training procedure.

EVIDENCE: Every tested additive-readout width from 128 through 118 has passed; the 118-unit model achieved 85.153% accuracy, making the next isolated one-channel reduction the most informative test of the recurrent-width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
=======
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
>>>>>>> REPLACE