MECHANISM: Boundary-seeking full-resolution GRU width reduction

HYPOTHESIS: A 103-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs than the qualified 104-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 103 while preserving full temporal coverage and the verified training procedure.

EVIDENCE: The 104-unit GRU narrowly qualified at 85.03%, while 102 units narrowly missed at 84.91%; testing 103 units directly resolves the remaining width boundary without repeating failed temporal subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
>>>>>>> REPLACE