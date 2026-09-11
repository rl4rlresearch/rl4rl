MECHANISM: Learned shared mel-subspace projection

HYPOTHESIS: A learned 20-to-16 feature projection feeding a 119-unit GRU will retain at least 85% accuracy while reducing exact dense MACs below the passing 118-unit, 23-step model.

INTENDED_EDIT: Add a bias-free learned input bottleneck, increase recurrent/readout width to 119, and preserve the verified 23-frame schedule and training procedure.

EVIDENCE: The 119-unit, 23-step model passed at 85.399%, providing more margin than the 118-unit model’s 85.031%; unlike the failed fixed 19-band averaging, a learned projection can preserve task-discriminative spectral combinations.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 16, bias=False)
        self.gru = nn.GRU(16, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
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