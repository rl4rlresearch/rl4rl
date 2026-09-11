MECHANISM: Early-frame pruning with recurrent capacity exchange

HYPOTHESIS: A 120-unit GRU using 22 causal steps will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.3% versus the passing 118-unit, 23-step model.

INTENDED_EDIT: Increase recurrent/readout width from 118 to 120 and remove the second currently processed early frame, preserving the first frame and all later coverage.

EVIDENCE: Increasing width by two while removing one step previously succeeded: the 119-unit, 23-step model reached 85.399% after the 117-unit, 23-step model missed narrowly, and it cost less than the passing 117-unit, 24-step design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            schedule[1:2] + schedule[3:-4] + schedule[-1:]
=======
            schedule[1:2] + schedule[4:-4] + schedule[-1:]
>>>>>>> REPLACE