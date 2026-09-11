MECHANISM: Full-resolution recurrent width bisection

HYPOTHESIS: A 104-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 112-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 104 while retaining full temporal coverage and the existing training procedure.

EVIDENCE: The 96-unit model narrowly failed at 84.29%, while the 112-unit model passed at 86.13%; testing their 104-unit midpoint directly refines the smallest viable full-resolution width after reduced-frame designs failed.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE