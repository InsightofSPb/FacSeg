"""Visualise predictions on facade datasets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from mmcv import Config
from mmseg.datasets import build_dataset

from lposs.models import build_model


def overlay(image, mask, alpha=0.5):
    color_mask = mask.copy()
    color_mask[mask == 1] = 255
    blended = image * (1 - alpha) + color_mask[:, :, None] * alpha
    return blended.astype(image.dtype)


def main(args):
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    dataset_cfg = Config.fromfile(cfg.evaluate.facade_damage.config)
    dataset_cfg.data.test.data_root = cfg.training.dataset.data_root
    dataset = build_dataset(dataset_cfg.data.test)
    class_names = dataset.CLASSES
    model = build_model(cfg.model, class_names=class_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(args.num_samples, len(dataset))):
        data = dataset[idx]
        img_container = data['img']
        if hasattr(img_container, 'data'):
            img_tensor = img_container.data
        else:
            img_tensor = img_container
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.forward_train(img_tensor)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        image = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = (image * 255).astype('uint8')
        blended = overlay(image, pred)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(blended)
        plt.title('Prediction overlay')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{idx:03d}.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise facade predictions")
    parser.add_argument('--config', default='facade_baseline.yaml')
    parser.add_argument('--output', default='./outputs/visualisations')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()
    main(args)
