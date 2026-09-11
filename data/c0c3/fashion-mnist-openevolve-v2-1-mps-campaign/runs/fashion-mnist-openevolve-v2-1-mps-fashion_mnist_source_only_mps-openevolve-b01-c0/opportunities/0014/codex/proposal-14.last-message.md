MECHANISM: Center-biased cardinal-translation augmentation

HYPOTHESIS: Sampling the centered view 50% of the time and each cardinal translation 12.5% will exceed 9,290 correct predictions by concentrating limited training exposure on the validation distribution while retaining useful one-pixel invariance.

INTENDED_EDIT: Reweight training augmentation from uniform sampling across five views to a 4:1:1:1:1 center-to-cardinal distribution; evaluation and all other settings remain unchanged.

EVIDENCE: Restricting augmentation from 25 translations to the centered and four cardinal views increased validation correct from 9,188 to 9,265, the largest recent gain; further concentrating exposure on the unshifted validation geometry directly extends that successful reduction in augmentation strength.

<<<<<<< SEARCH
    choices = torch.randint(
        0, cardinal_offsets.shape[0], (images.shape[0],), device=images.device
    )
    shifts = cardinal_offsets[choices]
=======
    choices = torch.randint(
        0, 8, (images.shape[0],), device=images.device
    )
    choices = torch.where(choices < 4, 0, choices - 3)
    shifts = cardinal_offsets[choices]
>>>>>>> REPLACE