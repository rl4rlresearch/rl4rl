MECHANISM: Incremental raw-skip upper-stage compression

HYPOTHESIS: Reducing only the upper GRU from 50 to 49 units will retain at least 85% validation accuracy while lowering total inference MACs from 633.3M to approximately 621.9M.

INTENDED_EDIT: Preserve the verified 49-unit lower GRU, folded 19-band input, and 28-step schedule, while reducing the upper recurrent state by one unit and adjusting the classifier input width.

EVIDENCE: The 49+50 hierarchy achieved 85.77% accuracy, while reducing the lower stage again failed at 84.54%; testing the uncompressed upper stage is the most direct alternative one-unit reduction and preserves the lower-stage capacity implicated by that failure.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(68, 50, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
=======
        self.upper_gru = nn.GRU(68, 49, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_hidden = torch.zeros(batch_size, 1, 50, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 50, device=device, dtype=dtype)
=======
        upper_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
>>>>>>> REPLACE