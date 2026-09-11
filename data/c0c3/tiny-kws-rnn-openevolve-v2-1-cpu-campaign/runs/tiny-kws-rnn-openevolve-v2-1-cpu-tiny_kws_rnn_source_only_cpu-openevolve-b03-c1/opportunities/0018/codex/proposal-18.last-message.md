MECHANISM: Conservative recurrent width reduction with trajectory readout

HYPOTHESIS: A 76-unit full-rate GRU using final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by 8.8% versus the successful 80-unit model.

INTENDED_EDIT: Reduce the GRU hidden width and both classifier input branches from 80 to 76 units while preserving all 32 causal frames and the established training procedure.

EVIDENCE: Successive four-unit reductions from 96 through 80 units all met the accuracy requirement; the 80-unit model achieved 85.64%, supporting one more structural width reduction without repeating failed temporal subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
=======
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(152, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
>>>>>>> REPLACE