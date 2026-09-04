MECHANISM: Full gate-feedback capacity with a one-unit state trim

HYPOTHESIS: A 100-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 333,211,120 to approximately 329,912,000.

INTENDED_EDIT: Reduce the recurrent state, input gate projections, gate expansions, temporal summary, and endpoint-augmented classifier from 101 to 100 units while preserving the qualified 16-frame schedule and training procedure.

EVIDENCE: The adjacent 101-unit design achieved 85.399% accuracy at 333,211,120 MACs, and each prior one-unit state trim from 103 through 101 qualified while saving exactly 3,299,120 MACs; keeping all three bottlenecks at 32 avoids the accuracy failures observed when feedback rank was reduced.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 303)
        self.recurrent_down = nn.Linear(101, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 101) for _ in range(3)]
        )
        self.classifier = nn.Linear(202, 8)
=======
        self.input_projection = nn.Linear(20, 300)
        self.recurrent_down = nn.Linear(100, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 100) for _ in range(3)]
        )
        self.classifier = nn.Linear(200, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE