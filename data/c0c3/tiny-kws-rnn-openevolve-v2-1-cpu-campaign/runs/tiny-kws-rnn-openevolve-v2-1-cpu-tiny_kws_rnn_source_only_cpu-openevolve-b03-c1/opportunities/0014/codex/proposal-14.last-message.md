MECHANISM: Conservative full-rate recurrent width reduction with trajectory readout

HYPOTHESIS: A 92-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the successful 96-unit model.

INTENDED_EDIT: Reduce the GRU hidden width and both classifier input branches from 96 to 92 units, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 96-unit endpoint-and-trajectory model achieved 86.26% accuracy with a 1.26-point margin, while temporal subsampling repeatedly failed; a modest width reduction preserves complete temporal evidence and directly targets the dominant recurrent MAC cost.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
>>>>>>> REPLACE