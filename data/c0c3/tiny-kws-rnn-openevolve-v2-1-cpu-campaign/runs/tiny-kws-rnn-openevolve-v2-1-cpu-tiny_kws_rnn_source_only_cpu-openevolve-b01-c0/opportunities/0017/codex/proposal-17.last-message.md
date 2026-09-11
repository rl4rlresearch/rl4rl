MECHANISM: Additive-readout recurrent-width trim

HYPOTHESIS: A 127-unit GRU with the passing 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.45%.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 127 while preserving the verified schedule and training procedure.

EVIDENCE: The current 128-unit additive-readout model reached 86.503% accuracy, 1.503 points above the threshold; additive fusion improved the same 25-step schedule from 84.540% with mean-only classification, providing evidence that the stronger readout may support the conservative one-unit width reduction that previously missed under the weaker architecture.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
>>>>>>> REPLACE