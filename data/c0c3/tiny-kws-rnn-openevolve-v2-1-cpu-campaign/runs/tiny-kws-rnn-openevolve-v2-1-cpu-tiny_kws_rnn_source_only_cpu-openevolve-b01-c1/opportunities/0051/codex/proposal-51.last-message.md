MECHANISM: One-unit upper-state contraction

HYPOTHESIS: Reducing only the upper GRU width from 48 to 47 will retain at least 85% validation accuracy while saving approximately 10.49M total inference MACs.

INTENDED_EDIT: Shrink the upper recurrent state by one unit and resize its state initialization and classifier input accordingly.

EVIDENCE: The current 49+48 hierarchy reaches 86.50%, while extensive upper-input compression retained accuracy and reducing the load-bearing lower width failed; this supports testing a small structural reduction in the more redundant upper stage.

<<<<<<< SEARCH
        self.upper_gru = nn.GRU(58, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
=======
        self.upper_gru = nn.GRU(58, 47, num_layers=1, batch_first=True)
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