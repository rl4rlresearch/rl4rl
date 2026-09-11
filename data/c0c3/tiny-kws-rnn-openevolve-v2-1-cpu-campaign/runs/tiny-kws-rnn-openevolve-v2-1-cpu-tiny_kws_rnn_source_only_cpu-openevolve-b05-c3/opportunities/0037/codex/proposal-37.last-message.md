MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 107-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 677,401,920 to approximately 665,897,380.

INTENDED_EDIT: Reduce the recurrent state, temporal summary, and both readout widths from 108 to 107 units while preserving the qualified 20-frame schedule and training procedure.

EVIDENCE: The adjacent 108-unit design achieved 87.48% validation accuracy at 677,401,920 MACs, leaving a 2.48-point margin after every tested 108–114-unit batch-64 GRU qualified; 107 units is the most informative remaining width-boundary probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)
        self.endpoint_classifier = nn.Linear(108, 8)
=======
        self.gru = nn.GRU(20, 107, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(107, 8)
        self.endpoint_classifier = nn.Linear(107, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 107, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 107, device=device, dtype=dtype)
>>>>>>> REPLACE