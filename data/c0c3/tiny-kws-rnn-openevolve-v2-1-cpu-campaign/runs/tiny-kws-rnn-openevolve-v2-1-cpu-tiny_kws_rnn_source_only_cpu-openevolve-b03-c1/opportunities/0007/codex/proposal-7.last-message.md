MECHANISM: Conservative full-rate recurrent width reduction

HYPOTHESIS: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the successful 104-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input from 104 to 100 units while preserving the full causal schedule and established training procedure.

EVIDENCE: The 104-unit full-rate GRU achieved 85.52% accuracy, and prior 112- and 108-unit models also cleared 85%; temporal subsampling repeatedly failed, so another small width reduction is the best-supported route to lower compute without discarding frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE