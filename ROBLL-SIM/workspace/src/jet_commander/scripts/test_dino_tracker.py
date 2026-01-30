#!/usr/bin/env python3
"""
Standalone DINOv2 tracker test script.
Tests tracking a reference image against a test image.
"""
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import argparse
import os

print(f"Numpy version: {np.__version__}")
print(f"Torch version: {torch.__version__}")


class DinoTrackerStandalone:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load DINOv2
        print("Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').eval().to(self.device)
        print("Model loaded")

        self.patch_size = 14
        self.ref_features = None

    def pil_to_tensor(self, img, size=518):
        """Convert PIL Image to tensor without numpy conversion issues."""
        img = img.convert('RGB')
        img = img.resize((size, size), Image.BILINEAR)

        img_data = list(img.getdata())
        width, height = img.size

        r = torch.tensor([p[0] for p in img_data], dtype=torch.float32).view(height, width) / 255.0
        g = torch.tensor([p[1] for p in img_data], dtype=torch.float32).view(height, width) / 255.0
        b = torch.tensor([p[2] for p in img_data], dtype=torch.float32).view(height, width) / 255.0

        img_tensor = torch.stack([r, g, b], dim=0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor.unsqueeze(0).to(self.device)

    def load_reference(self, path):
        """Load reference image and extract features."""
        print(f"Loading reference: {path}")
        img = Image.open(path).convert('RGB')
        img_tensor = self.pil_to_tensor(img)

        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_tokens = features['x_norm_patchtokens']

        # Average pool to get single reference vector
        self.ref_features = patch_tokens.mean(dim=1)
        self.ref_features = F.normalize(self.ref_features, dim=-1)

        print("Reference features extracted")
        return img

    def extract_features(self, img):
        """Extract patch-level features from image."""
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        img_tensor = self.pil_to_tensor(img)

        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_tokens = features['x_norm_patchtokens']

        return patch_tokens

    def find_match(self, img):
        """Find best match location in image."""
        if self.ref_features is None:
            raise ValueError("Load a reference image first!")

        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        orig_size = img.size  # (width, height)

        patch_features = self.extract_features(img)

        # Normalize
        patch_features_norm = F.normalize(patch_features, dim=-1)

        # Compute similarity
        similarity = torch.matmul(patch_features_norm, self.ref_features.transpose(-1, -2))
        similarity = similarity.squeeze(-1).squeeze(0)

        # Get grid dimensions
        num_patches = similarity.shape[0]
        grid_size = int(num_patches ** 0.5)

        # Reshape to grid
        sim_grid = similarity.view(grid_size, grid_size)

        # Find maximum
        max_val = sim_grid.max().item()
        max_idx = sim_grid.argmax().item()
        max_y = max_idx // grid_size
        max_x = max_idx % grid_size

        # Convert to pixel coordinates
        pixel_x = int((max_x + 0.5) / grid_size * orig_size[0])
        pixel_y = int((max_y + 0.5) / grid_size * orig_size[1])

        return pixel_x, pixel_y, max_val, sim_grid.cpu().numpy()

    def visualize(self, ref_img, test_img, save_path=None):
        """Visualize tracking result."""
        if isinstance(ref_img, str):
            ref_img = Image.open(ref_img).convert('RGB')
        if isinstance(test_img, str):
            test_img = Image.open(test_img).convert('RGB')

        # Find match
        px, py, conf, sim_grid = self.find_match(test_img)

        # Convert to numpy for visualization
        test_np = np.array(test_img)
        ref_np = np.array(ref_img)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Reference image
        axes[0, 0].imshow(ref_np)
        axes[0, 0].set_title("Reference Image")
        axes[0, 0].axis('off')

        # Test image with detection
        axes[0, 1].imshow(test_np)
        axes[0, 1].plot(px, py, 'r+', markersize=20, markeredgewidth=3)
        circle = plt.Circle((px, py), 30, color='lime', fill=False, linewidth=2)
        axes[0, 1].add_patch(circle)
        axes[0, 1].set_title(f"Detection (conf: {conf:.3f})")
        axes[0, 1].axis('off')

        # Similarity heatmap
        im = axes[1, 0].imshow(sim_grid, cmap='jet')
        axes[1, 0].set_title("Similarity Heatmap")
        plt.colorbar(im, ax=axes[1, 0])

        # Overlay
        sim_resized = cv2.resize(sim_grid, (test_np.shape[1], test_np.shape[0]))
        sim_norm = (sim_resized - sim_resized.min()) / (sim_resized.max() - sim_resized.min() + 1e-8)

        overlay = test_np.copy().astype(float) / 255.0
        heatmap = plt.cm.jet(sim_norm)[:, :, :3]
        blended = overlay * 0.6 + heatmap * 0.4
        blended = (blended * 255).astype(np.uint8)

        axes[1, 1].imshow(blended)
        axes[1, 1].plot(px, py, 'w+', markersize=20, markeredgewidth=3)
        axes[1, 1].set_title("Heatmap Overlay")
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

        return px, py, conf


def main():
    parser = argparse.ArgumentParser(description='Test DINOv2 tracker')
    parser.add_argument('--reference', '-r', required=True, help='Reference image path')
    parser.add_argument('--test', '-t', help='Test image path (optional)')
    parser.add_argument('--output', '-o', help='Output visualization path')
    args = parser.parse_args()

    tracker = DinoTrackerStandalone()

    # Load reference
    ref_img = tracker.load_reference(args.reference)

    if args.test:
        # Track in test image
        px, py, conf = tracker.visualize(ref_img, args.test, args.output)
        print(f"\nResult: Position ({px}, {py}), Confidence: {conf:.4f}")
    else:
        # Just show reference features
        print("No test image provided. Reference loaded successfully.")
        plt.imshow(ref_img)
        plt.title("Reference Image")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
