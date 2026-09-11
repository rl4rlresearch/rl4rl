MECHANISM: One-channel-width boundary probe

HYPOTHESIS: A 126-unit GRU on the verified 28-frame schedule will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 2.9% versus the passing 128-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 126 while preserving the successful training procedure and 28-frame schedule.

EVIDENCE: The 128-unit model passed at 85.276%, while 125 units narrowly missed at 84.908%; 126 units is the lowest untested width between these results and therefore the most informative next cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
>>>>>>> REPLACE