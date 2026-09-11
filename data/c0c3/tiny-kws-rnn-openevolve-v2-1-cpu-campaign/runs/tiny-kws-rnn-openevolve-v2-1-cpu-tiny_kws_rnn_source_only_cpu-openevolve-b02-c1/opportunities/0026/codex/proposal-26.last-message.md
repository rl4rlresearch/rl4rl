MECHANISM: Preserve-dominant-branch asymmetric width reduction

HYPOTHESIS: A 60+58 GRU with the verified orthonormal seven-coordinate classifier will retain at least 85% validation accuracy while reducing total inference MACs from 671,511,100 to approximately 661,791,410.

INTENDED_EDIT: Remove one hidden unit from the smaller GRU branch and resize its state, temporal summary, and classifier input from 119 to 118 features.

EVIDENCE: The 60+59 orthonormal model achieved 85.40% accuracy, whereas reducing the larger branch to form a symmetric 59+59 model failed at 84.29%; preserving the 60-unit branch tests whether the successful asymmetry is load-bearing while making the smallest remaining recurrent reduction.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 7)
=======
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
>>>>>>> REPLACE