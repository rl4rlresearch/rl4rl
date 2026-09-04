MECHANISM: Fine-grained recurrent-width scaling with dual temporal readout

HYPOTHESIS: A 68-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 9.6% versus the verified 72-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 72 to 68 and resize the classifier and recurrent summary state accordingly.

EVIDENCE: The 72-unit model achieved 86.13% accuracy after successive width reductions remained viable; testing the four-unit midpoint toward 64 is the most informative lower-risk continuation given its narrower 1.13-point margin.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 72, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(144, 8)
=======
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(136, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 72, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 72, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
>>>>>>> REPLACE