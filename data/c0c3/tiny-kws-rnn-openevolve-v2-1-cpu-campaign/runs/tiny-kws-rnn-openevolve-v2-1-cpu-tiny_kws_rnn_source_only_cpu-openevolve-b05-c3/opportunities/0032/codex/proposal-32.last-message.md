MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 112-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from the qualified 736,391,620 to approximately 724,398,080.

INTENDED_EDIT: Reduce the recurrent state and both readout widths from 115 to 112 units while preserving the qualified 20-frame schedule and batch-64 training procedure.

EVIDENCE: The adjacent 113-unit design achieved 86.63% accuracy at 736,391,620 MACs, leaving a 1.63-point margin; testing 112 units is the most informative remaining width-boundary probe under the proven batch-64 regime.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 115, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(115, 8)
        self.endpoint_classifier = nn.Linear(115, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
        self.endpoint_classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 115, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 115, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE