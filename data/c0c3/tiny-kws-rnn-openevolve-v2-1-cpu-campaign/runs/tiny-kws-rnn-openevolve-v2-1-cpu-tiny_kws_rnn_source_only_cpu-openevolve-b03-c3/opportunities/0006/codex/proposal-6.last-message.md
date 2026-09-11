MECHANISM: Conservative full-resolution GRU width reduction

HYPOTHESIS: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the qualified 104-unit design.

INTENDED_EDIT: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 104 to 100 while preserving the full-frame schedule and training procedure.

EVIDENCE: The 104-unit full-resolution GRU achieved 85.52% accuracy, and the preceding 112-to-108-to-104 width reductions all remained qualified; meanwhile, reducing temporal resolution to 24 or 16 steps produced less than 83% accuracy, motivating another four-unit width reduction while retaining every frame.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE