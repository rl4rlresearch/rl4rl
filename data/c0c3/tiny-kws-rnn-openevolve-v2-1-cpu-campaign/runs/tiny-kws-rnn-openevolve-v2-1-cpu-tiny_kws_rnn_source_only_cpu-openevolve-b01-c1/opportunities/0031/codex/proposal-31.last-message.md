MECHANISM: Raw-skip lower-stage width compression

HYPOTHESIS: Reducing only the lower GRU from 50 to 49 units will preserve at least 85% validation accuracy while lowering total inference MACs from 644.8M to approximately 633.3M.

INTENDED_EDIT: Keep the verified 28-step, folded-19-band hierarchy and 50-unit upper GRU, but reduce the lower recurrent state by one unit and adjust dependent dimensions.

EVIDENCE: The 50+50 raw-skip hierarchy achieved 85.77% accuracy and substantially outperformed the wider single-GRU alternatives; because the upper stage also receives normalized acoustic input directly, it can plausibly tolerate the smallest structural reduction in lower-stage width.

<<<<<<< SEARCH
        self.lower_gru = nn.GRU(19, 50, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(69, 50, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(200, 8)
=======
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(68, 50, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_hidden = torch.zeros(batch_size, 1, 50, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 50, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 50, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 50, device=device, dtype=dtype)
=======
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 50, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 50, device=device, dtype=dtype)
>>>>>>> REPLACE