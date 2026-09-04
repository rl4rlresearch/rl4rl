MECHANISM: One-unit recurrent-width reduction on the residual-pooled frontier

HYPOTHESIS: An 83-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 418,587,260.

INTENDED_EDIT: Reduce the GRU hidden state and classifier from 84 to 83 units while preserving the qualified 20-frame schedule, residual pooling, and training procedure.

EVIDENCE: The 84-unit model qualified at 85.64% and 427,738,080 MACs, while both preceding one-unit reductions from 86 to 85 and 85 to 84 retained qualification; 83 units is therefore the cheapest unresolved adjacent frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(84, 8)
=======
        self.gru = nn.GRU(20, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(83, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
>>>>>>> REPLACE