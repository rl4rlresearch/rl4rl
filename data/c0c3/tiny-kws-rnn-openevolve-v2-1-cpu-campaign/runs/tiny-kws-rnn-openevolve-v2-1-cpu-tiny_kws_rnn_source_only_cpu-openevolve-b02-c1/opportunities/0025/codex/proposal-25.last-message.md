MECHANISM: Step-to-width reallocation with orthonormal logit contrasts

HYPOTHESIS: A 61+60 GRU processing frames 4–31 with the verified orthonormal seven-coordinate classifier will recover the 0.46-point accuracy shortfall of the prior 28-step model, reaching at least 85% while reducing total inference MACs below the current 671,511,100.

INTENDED_EDIT: Skip one additional leading frame, widen the parallel GRUs from 60+59 to 61+60 units, resize the recurrent state and summary to 121 features, and retain the seven-coordinate orthonormal classifier.

EVIDENCE: The prior 61+60, 28-step model reached 84.54%, narrowly missing the threshold, while the orthonormal seven-coordinate classifier subsequently produced 85.40% on the verified 29-step architecture and reduced classifier MACs; this directly tests whether its balanced parameterization can close the small 28-step accuracy gap.

<<<<<<< SEARCH
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 7)
=======
        self.gru_a = nn.GRU(20, 61, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(121, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 61, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(3, available_frames))
=======
        return list(range(4, available_frames))
>>>>>>> REPLACE