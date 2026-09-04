MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 113-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 748,482,960 to approximately 736,391,620.

INTENDED_EDIT: Reduce the recurrent state and both readout widths from 114 to 113 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The adjacent 114-unit design achieved 86.01% accuracy at 748,482,960 MACs after the 115-unit design achieved 86.87%; this makes 113 units the most informative remaining width-boundary test under the proven batch-64 regime.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 114, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
        self.endpoint_classifier = nn.Linear(114, 8)
=======
        self.gru = nn.GRU(20, 113, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(113, 8)
        self.endpoint_classifier = nn.Linear(113, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 114, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 114, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 113, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 113, device=device, dtype=dtype)
>>>>>>> REPLACE