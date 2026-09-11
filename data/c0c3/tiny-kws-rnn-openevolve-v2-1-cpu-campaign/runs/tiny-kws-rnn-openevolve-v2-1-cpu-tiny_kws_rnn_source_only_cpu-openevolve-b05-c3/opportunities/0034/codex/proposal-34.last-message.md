MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 110-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 712,502,340 to approximately 700,704,400.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and both readout widths from 111 to 110 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The adjacent 111-unit design achieved 86.75% accuracy at 712,502,340 MACs, leaving a 1.75-point margin after every tested 111–114-unit batch-64 GRU qualified; 110 units is therefore the most informative remaining cost-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
        self.endpoint_classifier = nn.Linear(111, 8)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
        self.endpoint_classifier = nn.Linear(110, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE