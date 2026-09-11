MECHANISM: Learned input bottleneck with recurrent-capacity tradeoff

HYPOTHESIS: A learned 20-to-16 feature projection feeding a 100-unit GRU will retain at least 85% validation accuracy while reducing total dense MACs below the qualified 99-unit full-input GRU.

INTENDED_EDIT: Add a 16-dimensional linear frontend bottleneck and use the previously qualified 100-unit recurrent width throughout the state and classifier.

EVIDENCE: The full-input 100-unit GRU achieved 85.40% accuracy; replacing four input dimensions with a learned projection reduces estimated per-step MACs from 35,343 for the qualified 99-unit design to 35,120 while restoring one unit of recurrent capacity.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 16)
        self.gru = nn.GRU(16, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_projection(self.input_norm(frame)).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_projection(self.input_norm(frames)),
            hidden.transpose(0, 1).contiguous(),
>>>>>>> REPLACE