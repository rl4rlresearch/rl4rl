MECHANISM: Single-unit recurrent width refinement

HYPOTHESIS: An 88-unit dual-readout GRU over the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 712,728,910 to 698,265,920.

INTENDED_EDIT: Reduce the GRU hidden width from 89 to 88 and resize its state, temporal summary, and mean-plus-final classifier.

EVIDENCE: The 89-unit model achieved 86.26% accuracy, while every tested dense width from 90 through 96 also qualified; this 1.26-point margin supports the next conservative one-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 89, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(178, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 89, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 89, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE