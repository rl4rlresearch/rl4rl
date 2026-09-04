MECHANISM: Conservative recurrent width reduction

HYPOTHESIS: A 104-unit full-rate GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.7% versus the successful 108-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input from 108 to 104 units while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 108-unit full-rate model achieved 86.01% accuracy, while temporal subsampling repeatedly failed; the remaining accuracy margin supports another small structural width reduction without discarding temporal evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE