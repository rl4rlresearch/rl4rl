MECHANISM: Progressive adjacent high-frequency mel-band pooling

HYPOTHESIS: Independently averaging bands 14–15, 16–17, and 18–19 will retain at least 85% validation accuracy while reducing recurrent MACs by 7,503,705 versus the verified 18-input design.

INTENDED_EDIT: Reduce the GRU input width from 18 to 17 by retaining bands 0–13 and averaging the three highest adjacent mel-band pairs; preserve the 99-unit state, 31-step schedule, classifier, and training procedure.

EVIDENCE: Successive pooling reduced input width from 20 to 19 and then 18 while achieving 85.52% and 85.28% accuracy; the latest reduction cost only 0.24 percentage points, making one additional locality-preserving pair compression the most informative next structural-cost probe.

<<<<<<< SEARCH
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
=======
        self.gru = nn.GRU(17, 99, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        high_bands = normalized[..., 16:].reshape(
            *normalized.shape[:-1], 2, 2
        ).mean(dim=-1)
        return torch.cat((normalized[..., :16], high_bands), dim=-1)
=======
        high_bands = normalized[..., 14:].reshape(
            *normalized.shape[:-1], 3, 2
        ).mean(dim=-1)
        return torch.cat((normalized[..., :14], high_bands), dim=-1)
>>>>>>> REPLACE