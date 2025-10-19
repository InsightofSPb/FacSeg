"""Custom dataset definition for facade damage segmentation."""

from mmseg.datasets import DATASETS, CustomDataset


@DATASETS.register_module(force=True)
class FacadeDamageDataset(CustomDataset):
    """Facade damage dataset tailored for the facade baseline pipeline."""

    CLASSES = (
        "background",
        "CRACK",
        "SPALLING",
        "DELAMINATION",
        "MISSING_ELEMENT",
        "WATER_STAIN",
        "EFFLORESCENCE",
        "CORROSION",
        "ORNAMENT_INTACT",
        "REPAIRS",
        "TEXT_OR_IMAGES",
    )

    # Default palette roughly matches the Label Studio colours shared by the user.
    PALETTE = [
        [0, 0, 0],
        [229, 57, 53],
        [30, 136, 229],
        [67, 160, 71],
        [251, 140, 0],
        [142, 36, 170],
        [253, 216, 53],
        [0, 172, 193],
        [158, 158, 158],
        [78, 158, 158],
        [142, 126, 71],
    ]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

        classes = kwargs.get('classes')
        if classes is not None:
            self.CLASSES = tuple(classes)
        palette = kwargs.get('palette')
        if palette is not None:
            self.PALETTE = [list(color) for color in palette]
