MECHANISM: Diagonal-to-center translation redistribution

HYPOTHESIS: Reassigning diagonal one-pixel training shifts to the centered crop will exceed 9,196 correct predictions by eliminating transforms unsupported by the successful validation ensemble while preserving the baseline frequency of each cardinal shift.

INTENDED_EDIT: Keep the existing triangular ±1 sampling, but convert samples shifted on both axes to centered crops, changing the distribution from 25% center/50% cardinal/25% diagonal to 50% center/50% cardinal/0% diagonal.

EVIDENCE: Restricting translations from ±2 to ±1 improved correctness from 9,166 to 9,196, while diagonal validation views previously reduced correctness to 9,155. Unlike the timed-out 3:2:2:2:2 proposal, this isolates diagonal removal without increasing total cardinal-shift exposure.

<<<<<<< SEARCH
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
=======
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    diagonal_mask = (offset_y != 2) & (offset_x != 2)
    offset_y = torch.where(diagonal_mask, torch.full_like(offset_y, 2), offset_y)
    offset_x = torch.where(diagonal_mask, torch.full_like(offset_x, 2), offset_x)
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
>>>>>>> REPLACE