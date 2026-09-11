MECHANISM: Full-resolution recurrent width bisection

HYPOTHESIS: A 100-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 104-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 100 while retaining full temporal coverage and the existing training procedure.

EVIDENCE: The 96-unit model narrowly failed at 84.29%, while the 104-unit model passed at 85.89%; testing their 100-unit midpoint most directly refines the minimum viable recurrent width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE