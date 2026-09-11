MECHANISM: Incremental raw-skip lower-stage compression

HYPOTHESIS: Reducing the lower GRU from 49 to 48 units will preserve at least 85% validation accuracy while lowering total inference MACs by approximately 11.4M from 633.3M.

INTENDED_EDIT: Reduce the lower recurrent state by one unit and update the upper-stage and classifier input dimensions accordingly.

EVIDENCE: The previous 50-to-49 lower-stage reduction preserved exactly 85.77% accuracy while reducing total MACs by 11.5M, indicating that the raw acoustic skip lets the upper GRU tolerate incremental compression of the lower stage.

<<<<<<< SEARCH
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(68, 50, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
=======
        self.lower_gru = nn.GRU(19, 48, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(67, 50, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 50, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
=======
        lower_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 50, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
>>>>>>> REPLACE