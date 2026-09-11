MECHANISM: Full-resolution one-unit recurrent width refinement

HYPOTHESIS: A 99-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 100-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 99 while preserving full temporal coverage and training.

EVIDENCE: The 100-unit model passed at 85.40%, while the 98-unit model failed at 84.54%; testing 99 directly resolves the remaining width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE