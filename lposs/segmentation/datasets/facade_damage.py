"""Custom dataset definition for facade damage segmentation."""

from mmseg.datasets import DATASETS, CustomDataset


@DATASETS.register_module(force=True)
class FacadeDamageDataset(CustomDataset):
    """Facade damage dataset with two default classes."""

    CLASSES = ("background", "damage")
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

        classes = kwargs.get('classes')
        if classes is not None:
            self.CLASSES = tuple(classes)
        palette = kwargs.get('palette')
        if palette is not None:
            self.PALETTE = [list(color) for color in palette]
