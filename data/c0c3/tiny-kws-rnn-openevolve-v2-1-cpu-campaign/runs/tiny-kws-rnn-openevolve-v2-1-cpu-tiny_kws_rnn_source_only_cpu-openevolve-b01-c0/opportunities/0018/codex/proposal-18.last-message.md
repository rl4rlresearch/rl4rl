MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 126-unit GRU with the passing 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.46% versus the verified 127-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 127 to 126 while preserving the successful schedule, additive readout, and training procedure.

EVIDENCE: The 127-unit additive-readout model achieved 86.380% accuracy, 1.380 percentage points above the requirement; this supports testing the smallest remaining structural width reduction without changing its verified temporal coverage or readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
=======
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
>>>>>>> REPLACE