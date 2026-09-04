MECHANISM: One-unit recurrent width contraction

HYPOTHESIS: Reducing the GRU width from 58 to 57 while preserving the verified 23-frame schedule will retain at least 85% validation accuracy and reduce total inference MACs from 255,163,460 to approximately 247,558,695.

INTENDED_EDIT: Shrink the GRU hidden state by one unit and adjust the temporal summary and classifier dimensions accordingly.

EVIDENCE: The current 23-step, width-58 model achieved 85.64%, while reducing temporal coverage to 22 steps failed at 84.66%; preserving all successful frame selections and testing the smallest recurrent-width reduction is the best-supported orthogonal cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
=======
        self.gru = nn.GRU(20, 57, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 57, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 57, device=device, dtype=dtype)
>>>>>>> REPLACE