MECHANISM: Preserve temporal coverage while trimming recurrent state width

HYPOTHESIS: A 125-unit GRU processing the verified 28-frame schedule will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 4.3%, below even the failed 27-step 128-unit design.

INTENDED_EDIT: Reduce the GRU hidden state and classifier input width from 128 to 125 without changing training or the successful 28-frame schedule.

EVIDENCE: The 128-unit model passed at 28 steps with 85.276% accuracy but failed at 27 steps with 84.294%, indicating temporal evidence is sensitive; retaining all 28 observations while making a modest width reduction targets quadratic recurrent cost instead.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
>>>>>>> REPLACE