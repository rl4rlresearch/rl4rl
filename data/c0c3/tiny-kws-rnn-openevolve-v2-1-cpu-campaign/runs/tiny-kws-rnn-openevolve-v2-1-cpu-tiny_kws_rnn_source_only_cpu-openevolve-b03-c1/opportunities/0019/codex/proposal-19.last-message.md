MECHANISM: Boundary-seeking recurrent width interpolation

HYPOTHESIS: A 78-unit full-rate GRU with endpoint-and-trajectory readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 4.5% versus the successful 80-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 80 to 78 units and resize its state, temporal summary, and classifier input accordingly, while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 80-unit model achieved 85.64% accuracy, while the 76-unit model narrowly missed at 84.79%; testing the midpoint directly targets the apparent capacity boundary without discarding temporal evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
=======
        self.gru = nn.GRU(20, 78, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(156, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 78, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
>>>>>>> REPLACE