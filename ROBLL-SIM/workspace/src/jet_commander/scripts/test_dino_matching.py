#!/usr/bin/env python3
"""
Test script to visualize DINOv2 feature matching between reference and test images.
Uses patch-to-patch matching with max pooling to focus on object features.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


def pil_to_tensor(img, device, size=518):
    """Convert PIL Image to tensor."""
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

    return img_tensor.unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description='Test DINOv2 feature matching')
    parser.add_argument('--ref', default='/workspace/src/jet_commander/images/image.png',
                        help='Reference image path')
    parser.add_argument('--test', default='/workspace/src/jet_commander/images/test_image.png',
                        help='Test image path')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--output', default='/workspace/src/jet_commander/images/match_result.png',
                        help='Output visualization path')
    parser.add_argument('--method', default='max_pool', choices=['average', 'max_pool', 'center'],
                        help='Feature matching method')
    args = parser.parse_args()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DINOv2
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').eval().to(device)
    print("Model loaded")

    # Load images
    print(f"Loading reference image: {args.ref}")
    ref_img = Image.open(args.ref).convert('RGB')
    print(f"Loading test image: {args.test}")
    test_img = Image.open(args.test).convert('RGB')

    # Convert to tensors
    ref_tensor = pil_to_tensor(ref_img, device)
    test_tensor = pil_to_tensor(test_img, device)

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        # Reference patch features
        ref_features = model.forward_features(ref_tensor)
        ref_patch_tokens = ref_features['x_norm_patchtokens']  # [1, N, D]
        ref_patch_norm = F.normalize(ref_patch_tokens, dim=-1)

        # Test image patch features
        test_features = model.forward_features(test_tensor)
        test_patch_tokens = test_features['x_norm_patchtokens']  # [1, N, D]
        test_patch_norm = F.normalize(test_patch_tokens, dim=-1)

    # Get grid dimensions
    num_patches = ref_patch_tokens.shape[1]
    grid_size = int(num_patches ** 0.5)
    print(f"Grid size: {grid_size}x{grid_size} ({num_patches} patches)")

    # Compute similarity based on method
    print(f"Computing similarity using method: {args.method}")
    match_indices = None  # Will store which ref patch each test patch matched to

    if args.method == 'average':
        # Original method: average all reference patches
        ref_vector = ref_patch_norm.mean(dim=1)
        ref_vector = F.normalize(ref_vector, dim=-1)
        similarity = torch.matmul(test_patch_norm, ref_vector.transpose(-1, -2))
        similarity = similarity.squeeze(-1).squeeze(0)
        method_desc = "Average of all ref patches"

    elif args.method == 'max_pool':
        # Better method: for each test patch, find max similarity with any ref patch
        # This finds patches in test that match ANY part of the reference
        # Shape: [1, N_test, D] x [1, D, N_ref] -> [1, N_test, N_ref]
        all_similarities = torch.matmul(test_patch_norm, ref_patch_norm.transpose(-1, -2))
        # Take max over reference patches for each test patch
        similarity, match_indices = all_similarities.max(dim=-1)  # [1, N_test], [1, N_test]
        similarity = similarity.squeeze(0)
        match_indices = match_indices.squeeze(0).cpu().numpy()  # Which ref patch each test patch matched
        method_desc = "Max similarity with any ref patch"

    elif args.method == 'center':
        # Use only center region of reference (where the jet is)
        ref_grid = ref_patch_norm.view(1, grid_size, grid_size, -1)
        # Extract center 50% of patches
        margin = grid_size // 4
        center_patches = ref_grid[:, margin:grid_size-margin, margin:grid_size-margin, :]
        center_patches = center_patches.reshape(1, -1, ref_patch_norm.shape[-1])
        # Average center patches
        ref_vector = center_patches.mean(dim=1)
        ref_vector = F.normalize(ref_vector, dim=-1)
        similarity = torch.matmul(test_patch_norm, ref_vector.transpose(-1, -2))
        similarity = similarity.squeeze(-1).squeeze(0)
        method_desc = f"Average of center {(grid_size-2*margin)**2} patches"

    # Reshape to grid
    sim_grid = similarity.view(grid_size, grid_size).cpu().numpy()

    # Find maximum
    max_val = sim_grid.max()
    max_idx = sim_grid.argmax()
    max_y = max_idx // grid_size
    max_x = max_idx % grid_size

    # Convert to normalized coordinates
    norm_x = (max_x + 0.5) / grid_size
    norm_y = (max_y + 0.5) / grid_size

    print(f"\nMethod: {method_desc}")
    print(f"\nResults:")
    print(f"  Max similarity: {max_val:.4f}")
    print(f"  Grid position: ({max_x}, {max_y})")
    print(f"  Normalized position: ({norm_x:.3f}, {norm_y:.3f})")
    print(f"  Above threshold ({args.threshold}): {max_val > args.threshold}")

    # Statistics
    print(f"\nSimilarity statistics:")
    print(f"  Min: {sim_grid.min():.4f}")
    print(f"  Max: {sim_grid.max():.4f}")
    print(f"  Mean: {sim_grid.mean():.4f}")
    print(f"  Std: {sim_grid.std():.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'DINOv2 Feature Matching - Method: {args.method}', fontsize=14)

    # Reference image
    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')

    # Test image with detection
    test_np = np.array(test_img)
    orig_h, orig_w = test_np.shape[:2]
    pixel_x = int(norm_x * orig_w)
    pixel_y = int(norm_y * orig_h)

    test_marked = test_np.copy()
    color = (0, 255, 0) if max_val > args.threshold else (255, 0, 0)
    cv2.circle(test_marked, (pixel_x, pixel_y), 15, color, 3)
    cv2.line(test_marked, (pixel_x - 25, pixel_y), (pixel_x + 25, pixel_y), color, 3)
    cv2.line(test_marked, (pixel_x, pixel_y - 25), (pixel_x, pixel_y + 25), color, 3)

    axes[0, 1].imshow(test_marked)
    axes[0, 1].set_title(f'Test Image (conf: {max_val:.3f})')
    axes[0, 1].axis('off')

    # Similarity heatmap (raw)
    im = axes[1, 0].imshow(sim_grid, cmap='hot', interpolation='nearest')
    axes[1, 0].scatter([max_x], [max_y], c='cyan', s=100, marker='x', linewidths=3)
    axes[1, 0].set_title(f'Similarity Grid ({grid_size}x{grid_size})')
    plt.colorbar(im, ax=axes[1, 0])

    # Similarity heatmap overlaid on test image
    sim_resized = cv2.resize(sim_grid, (orig_w, orig_h))
    sim_normalized = ((sim_resized - sim_resized.min()) /
                      (sim_resized.max() - sim_resized.min() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sim_normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(test_np, 0.5, heatmap, 0.5, 0)

    # Add crosshair
    cv2.circle(overlay, (pixel_x, pixel_y), 15, color, 3)
    cv2.line(overlay, (pixel_x - 25, pixel_y), (pixel_x + 25, pixel_y), color, 3)
    cv2.line(overlay, (pixel_x, pixel_y - 25), (pixel_x, pixel_y + 25), color, 3)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Heatmap Overlay')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nVisualization saved to: {args.output}")

    # Create matching lines visualization (only for max_pool method)
    if match_indices is not None:
        # Create side-by-side image for drawing matches
        ref_np = np.array(ref_img)
        ref_h, ref_w = ref_np.shape[:2]

        # Resize both images to same height for side-by-side display
        target_h = max(ref_h, orig_h)
        ref_display = cv2.resize(ref_np, (int(ref_w * target_h / ref_h), target_h))
        test_display = cv2.resize(test_np, (int(orig_w * target_h / orig_h), target_h))

        ref_disp_h, ref_disp_w = ref_display.shape[:2]
        test_disp_h, test_disp_w = test_display.shape[:2]

        # Create combined image
        combined = np.zeros((target_h, ref_disp_w + test_disp_w, 3), dtype=np.uint8)
        combined[:ref_disp_h, :ref_disp_w] = ref_display
        combined[:test_disp_h, ref_disp_w:] = test_display

        # Get top N matches above threshold
        sim_flat = sim_grid.flatten()
        top_k = min(20, (sim_flat > args.threshold).sum())  # Top 20 or all above threshold

        if top_k > 0:
            top_indices = np.argsort(sim_flat)[-top_k:][::-1]  # Best matches first

            # Generate colors for each match line
            colors = plt.cm.rainbow(np.linspace(0, 1, top_k))

            for i, test_idx in enumerate(top_indices):
                test_y = test_idx // grid_size
                test_x = test_idx % grid_size
                ref_idx = match_indices[test_idx]
                ref_y = ref_idx // grid_size
                ref_x = ref_idx % grid_size
                conf = sim_flat[test_idx]

                # Convert grid coords to pixel coords in display images
                ref_px = int((ref_x + 0.5) / grid_size * ref_disp_w)
                ref_py = int((ref_y + 0.5) / grid_size * ref_disp_h)
                test_px = int((test_x + 0.5) / grid_size * test_disp_w) + ref_disp_w
                test_py = int((test_y + 0.5) / grid_size * test_disp_h)

                # Draw line and points
                color_bgr = (int(colors[i][2]*255), int(colors[i][1]*255), int(colors[i][0]*255))
                cv2.line(combined, (ref_px, ref_py), (test_px, test_py), color_bgr, 2)
                cv2.circle(combined, (ref_px, ref_py), 5, color_bgr, -1)
                cv2.circle(combined, (test_px, test_py), 5, color_bgr, -1)

            # Add labels
            cv2.putText(combined, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Test", (ref_disp_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Top {top_k} matches (threshold={args.threshold})",
                       (10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save matching lines visualization
        match_output = args.output.replace('.png', '_matches.png')
        fig2, ax2 = plt.subplots(figsize=(16, 8))
        ax2.imshow(combined)
        ax2.set_title(f'Feature Matching Lines - Top {top_k} matches (method: {args.method})')
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(match_output, dpi=150)
        print(f"Matching lines saved to: {match_output}")

    # Also show if running interactively
    plt.show()


if __name__ == '__main__':
    main()
