MECHANISM: Incremental upper-stage width compression

HYPOTHESIS: Reducing the upper GRU from 48 to 47 units will preserve at least 85% validation accuracy while lowering exact inference MACs below the verified 610.56M, because the preceding 49-to-48 reduction retained 86.01% accuracy.

INTENDED_EDIT: Preserve the 49-unit lower GRU, folded 19-band input, 28-step schedule, summaries, and training procedure while reducing the upper recurrent state by one unit and adjusting its classifier dimensions.

EVIDENCE: The 49+48 hierarchy achieved 86.01% accuracy, leaving 1.01 percentage points of margin, while the failed 48+50 result indicates the lower stage is more sensitive to compression than the upper stage.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(68, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
=======
        self.upper_gru = nn.GRU(68, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
=======
        upper_hidden = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 47, device=device, dtype=dtype)
>>>>>>> REPLACE