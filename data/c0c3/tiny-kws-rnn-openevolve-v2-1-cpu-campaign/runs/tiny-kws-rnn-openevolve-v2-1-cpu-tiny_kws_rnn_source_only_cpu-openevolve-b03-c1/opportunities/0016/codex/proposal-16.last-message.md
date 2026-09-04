MECHANISM: Conservative recurrent width reduction with trajectory readout

HYPOTHESIS: An 84-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.1% versus the successful 88-unit model.

INTENDED_EDIT: Reduce the GRU hidden width and both classifier input branches from 88 to 84 units while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The endpoint-and-trajectory design remained above the requirement through successive reductions from 96 to 92 to 88 units, with the 88-unit model achieving 85.77%; another four-unit reduction directly tests the remaining capacity margin without repeating the failed temporal-subsampling strategy.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE