MECHANISM: Paired high-frequency spectral folding

HYPOTHESIS: Averaging bands 16–17 and 18–19 into two features will preserve at least 85% validation accuracy while reducing total inference MACs from 699.7M to approximately 693.4M.

INTENDED_EDIT: Reduce the GRU input width from 19 to 18 and replace the existing single high-band fold with two adjacent high-band averages in both recurrent execution paths.

EVIDENCE: Folding bands 18–19 recovered accuracy to 85.28% where deleting the highest band reached only 84.91%, showing that fixed adjacent-band aggregation can preserve useful spectral information; extending that mechanism to the neighboring high-frequency pair is the most evidence-supported remaining structural reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 92, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 92, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.cat(
            (
                frames[..., :18],
                frames[..., 18:20].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
=======
        return torch.cat(
            (
                frames[..., :16],
                frames[..., 16:18].mean(dim=-1, keepdim=True),
                frames[..., 18:20].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
>>>>>>> REPLACE