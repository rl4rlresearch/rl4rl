MECHANISM: Zero-initialized positional self-attention over fused spatial features

HYPOTHESIS: Adding content-dependent interactions between all 7×7 feature locations will exceed 9,328 correct predictions by modeling relationships between garment parts that the existing local convolutions and static flattened classifier cannot express efficiently.

INTENDED_EDIT: Preserve the successful mirrored-view fusion, but insert a four-head positional self-attention residual before classification. Its output projection starts at zero so training begins as the verified baseline while learning a genuinely nonlocal prediction mechanism.

EVIDENCE: Dense-head widening reduced performance to 9,300 and disagreement-only fusion reached 9,290, so neither generic classifier capacity nor removing the invariant pathway helped. This keeps that pathway and instead challenges the shared assumption that purely local features followed by fixed dense aggregation are sufficient.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        self.spatial_position = nn.Parameter(torch.zeros(1, 7 * 7, 64))
        self.spatial_norm = nn.LayerNorm(64)
        self.spatial_attention = nn.MultiheadAttention(
            64,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
            self.spatial_attention.out_proj.weight.zero_()
            self.spatial_attention.out_proj.bias.zero_()
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        tokens = fused.flatten(2).transpose(1, 2)
        positioned = self.spatial_norm(tokens + self.spatial_position)
        context, _ = self.spatial_attention(
            positioned,
            positioned,
            positioned,
            need_weights=False,
        )
        attended = (tokens + context).transpose(1, 2).reshape_as(fused)
        return self.classifier(attended)
>>>>>>> REPLACE