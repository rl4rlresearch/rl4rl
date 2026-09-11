MECHANISM: Full-resolution recurrent width bisection

HYPOTHESIS: A 98-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 100-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 98 while preserving full temporal coverage and the existing training procedure.

EVIDENCE: The 100-unit model passed at 85.40% while the 96-unit model failed at 84.29%; testing the 98-unit midpoint most directly refines the minimum viable recurrent width.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE