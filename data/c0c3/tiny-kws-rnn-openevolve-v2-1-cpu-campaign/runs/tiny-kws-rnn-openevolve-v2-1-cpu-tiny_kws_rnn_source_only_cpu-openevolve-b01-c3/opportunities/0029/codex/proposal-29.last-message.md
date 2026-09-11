MECHANISM: Endpoint-augmented one-unit width trim

HYPOTHESIS: A 93-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 420,435,680 to approximately 412,324,800.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier input width from 94 to 93 units while preserving the qualified 16-frame schedule and training procedure.

EVIDENCE: The adjacent 94-unit design achieved 87.239% accuracy at 420,435,680 MACs, leaving a 2.239-point margin and making 93 units the most informative next structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)
=======
        self.gru = nn.GRU(20, 93, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(186, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 93, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 93, device=device, dtype=dtype)
>>>>>>> REPLACE