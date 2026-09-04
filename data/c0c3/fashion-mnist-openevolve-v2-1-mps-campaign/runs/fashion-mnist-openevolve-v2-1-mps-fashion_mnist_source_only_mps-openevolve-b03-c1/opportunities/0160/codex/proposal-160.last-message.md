MECHANISM: Axial silhouette context channels

HYPOTHESIS: Providing broadcast row and column intensity profiles before the first convolution will exceed 9,210 correct predictions by exposing whole-image silhouette information before repeated downsampling.

INTENDED_EDIT: Add two parameter-free axial projection channels, expand the first convolution to six inputs, and reduce the channel-gate bottleneck to 22 units, keeping the model below 250,000 parameters.

EVIDENCE: Global spatial classification fell to 9,128, showing spatial layout is load-bearing, while augmentation and multi-view pooling improved results to 9,210, showing unresolved alignment sensitivity. Axial profiles preserve the successful flattened spatial head while giving it shift-responsive global shape context without another costly pooling mechanism.

<<<<<<< SEARCH
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(6, 24, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.channel_gate = nn.Sequential(
            nn.Linear(96, 24),
            nn.GELU(),
            nn.Linear(24, 96),
        )
=======
        self.channel_gate = nn.Sequential(
            nn.Linear(96, 22),
            nn.GELU(),
            nn.Linear(22, 96),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        details = F.conv2d(padded, detail_kernels)
        represented = torch.cat((images, details), dim=1)
        features = self.features(represented)
=======
        details = F.conv2d(padded, detail_kernels)
        row_profile = images.mean(dim=3, keepdim=True).expand_as(images)
        column_profile = images.mean(dim=2, keepdim=True).expand_as(images)
        represented = torch.cat(
            (images, details, row_profile, column_profile), dim=1
        )
        features = self.features(represented)
>>>>>>> REPLACE