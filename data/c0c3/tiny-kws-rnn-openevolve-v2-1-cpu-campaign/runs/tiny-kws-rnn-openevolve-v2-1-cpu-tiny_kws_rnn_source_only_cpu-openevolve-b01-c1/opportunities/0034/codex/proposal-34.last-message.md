MECHANISM: Incremental upper-stage width compression

HYPOTHESIS: Reducing only the upper GRU from 49 to 48 units will preserve at least 85% validation accuracy while lowering total inference MACs, because the verified 49+49 model reached 86.50% and prior failure from reducing the lower stage indicates lower-stage capacity is more load-bearing.

INTENDED_EDIT: Preserve the 49-unit lower GRU, folded 19-band input, and 28-step schedule while reducing the upper recurrent state to 48 units and adjusting its state tensors and classifier width.

EVIDENCE: The current 49+49 hierarchy achieved 86.50%, providing 1.50 percentage points of margin; by contrast, the 48+50 hierarchy achieved only 84.54%, motivating compression of the upper rather than lower stage.

<<<<<<< SEARCH
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(68, 49, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
=======
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(68, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        upper_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
=======
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
>>>>>>> REPLACE