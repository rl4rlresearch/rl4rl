MECHANISM: Low-rank endpoint residual readout

HYPOTHESIS: A 116-unit GRU with a rank-4 endpoint correction will retain at least 85% validation accuracy while reducing total inference MACs from the qualified 772,959,040 to approximately 772,606,960.

INTENDED_EDIT: Reduce the current GRU from 120 to the proven 116 units and factor the endpoint classifier through a four-unit linear bottleneck, preserving its zero-initialized residual behavior.

EVIDENCE: The 116-unit full endpoint-readout model achieved 85.77% accuracy, whereas 115 units failed at 84.17%; retaining the proven recurrent width while compressing only the complementary endpoint correction is the lower-risk next structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
        self.endpoint_classifier = nn.Linear(120, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)
=======
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
        self.endpoint_classifier = nn.Sequential(
            nn.Linear(116, 4, bias=False),
            nn.Linear(4, 8),
        )
        nn.init.zeros_(self.endpoint_classifier[1].weight)
        nn.init.zeros_(self.endpoint_classifier[1].bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
>>>>>>> REPLACE