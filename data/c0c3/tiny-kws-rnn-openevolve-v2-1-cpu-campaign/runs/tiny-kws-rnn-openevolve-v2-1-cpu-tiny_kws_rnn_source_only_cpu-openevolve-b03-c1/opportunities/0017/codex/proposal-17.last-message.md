MECHANISM: Conservative recurrent width reduction with trajectory readout

HYPOTHESIS: An 80-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the successful 84-unit model.

INTENDED_EDIT: Reduce the GRU hidden width and both classifier input branches from 84 to 80 units while preserving all 32 causal frames and the established training procedure.

EVIDENCE: Successive endpoint-and-trajectory GRU reductions from 96 to 92 to 88 to 84 units all exceeded 85% accuracy; the 84-unit model achieved 85.64%, supporting another four-unit structural reduction without discarding temporal evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
>>>>>>> REPLACE