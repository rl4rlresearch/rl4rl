MECHANISM: Full-rate recurrent width reduction

HYPOTHESIS: A 112-unit GRU processing all 32 frames will preserve at least 85% accuracy while reducing recurrent MACs by approximately 22% versus the successful 128-unit baseline.

INTENDED_EDIT: Reduce the GRU hidden state and classifier input from 128 to 112 units while retaining the complete causal frame schedule.

EVIDENCE: Temporal reduction was costly—28 frames achieved only 83.93%—while the 32-frame baseline achieved 87.24%; preserving temporal evidence and reducing recurrent width targets compute without repeating the demonstrated failure mode.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE