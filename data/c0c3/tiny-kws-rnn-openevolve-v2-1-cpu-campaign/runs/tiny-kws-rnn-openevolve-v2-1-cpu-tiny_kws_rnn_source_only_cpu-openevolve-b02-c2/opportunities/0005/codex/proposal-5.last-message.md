MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the qualified 112-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 104 while preserving the proven full-frame schedule and training procedure.

EVIDENCE: The 112-unit full-resolution GRU achieved 86.13% accuracy and improved cost over the qualified 120-unit model, while reduced-frame designs failed; another eight-unit width reduction is the most direct test of remaining capacity margin.

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