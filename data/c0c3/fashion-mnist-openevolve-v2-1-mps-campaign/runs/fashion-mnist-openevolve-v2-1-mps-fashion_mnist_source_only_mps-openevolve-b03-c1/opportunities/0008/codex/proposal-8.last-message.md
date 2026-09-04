MECHANISM: Center-weighted per-image translation augmentation

HYPOTHESIS: Biasing training crops toward zero- and one-pixel translations while retaining occasional two-pixel shifts will exceed 9,082 correct predictions.

INTENDED_EDIT: Replace uniform ±2-pixel offsets with independent triangular offsets formed by summing two uniform three-way samples.

EVIDENCE: Ten-view inference using central and one-pixel translations reached 9,082 correct, whereas uniformly averaging the full ±2 support fell to 9,074, indicating that extreme transformations should receive less weight.

<<<<<<< SEARCH
    offset_y = torch.randint(0, 5, (images.shape[0],), device=images.device)
    offset_x = torch.randint(0, 5, (images.shape[0],), device=images.device)
=======
    offset_y = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    offset_x = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
>>>>>>> REPLACE