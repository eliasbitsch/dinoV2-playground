#!/usr/bin/env python3
"""
DINOv2-based object tracker for ROS2.
Tracks a reference image in camera frames using DINOv2 feature matching.
"""
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2


class DinoTracker(Node):
    def __init__(self):
        super().__init__('dino_tracker')

        # Parameters
        self.declare_parameter('reference_image', '/workspace/src/jet_commander/images/ref_image_NBG.png')
        self.declare_parameter('camera_topic', '/front_camera')
        self.declare_parameter('patch_size', 14)
        self.declare_parameter('threshold', 0.70)
        self.declare_parameter('search_top_half', False)  # False = search bottom half of image

        ref_image_path = self.get_parameter('reference_image').value
        camera_topic = self.get_parameter('camera_topic').value
        self.patch_size = self.get_parameter('patch_size').value
        self.threshold = self.get_parameter('threshold').value
        self.search_top_half = self.get_parameter('search_top_half').value

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load DINOv2
        self.get_logger().info("Loading DINOv2 model...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').eval().to(self.device)
        self.get_logger().info("DINOv2 model loaded")

        # Load and process reference image
        self.get_logger().info(f"Loading reference image: {ref_image_path}")
        self.ref_features = self._load_reference(ref_image_path)
        self.get_logger().info("Reference features extracted")

        # ROS setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            RosImage,
            camera_topic,
            self.image_callback,
            10
        )

        # Publisher for tracked position
        self.position_pub = self.create_publisher(Point, '/tracked_position', 10)

        # Publisher for debug image
        self.debug_pub = self.create_publisher(RosImage, '/dino_debug', 10)

        # Publisher for reference image (for visualization)
        self.ref_pub = self.create_publisher(RosImage, '/dino_reference', 10)

        # Load and publish reference image once
        self._publish_reference_image(ref_image_path)

        self.get_logger().info(f"Subscribed to {camera_topic}")

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

    def _load_reference(self, path):
        """Load reference image and extract patch-level features."""
        img = Image.open(path).convert('RGB')
        img_tensor = self.pil_to_tensor(img)

        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_tokens = features['x_norm_patchtokens']

        # Keep all patch tokens (normalized) for patch-to-patch matching
        # This allows matching against any part of the reference, not just average
        ref_patches = F.normalize(patch_tokens, dim=-1)

        return ref_patches

    def _publish_reference_image(self, path):
        """Load and publish reference image for visualization."""
        img = Image.open(path).convert('RGB')
        img_array = np.array(img)
        ref_msg = self.bridge.cv2_to_imgmsg(img_array, encoding='rgb8')

        # Create a timer to periodically publish the reference image
        def publish_ref():
            self.ref_pub.publish(ref_msg)

        self.ref_timer = self.create_timer(1.0, publish_ref)  # Publish every 1 second

    def extract_patch_features(self, img_tensor):
        """Extract patch-level features from image."""
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_tokens = features['x_norm_patchtokens']

        return patch_tokens

    def find_best_match(self, patch_features, center_margin=0.2):
        """Find the best matching patch location using max-pooled cosine similarity.

        Uses blur and morphological operations to clean up the similarity map.

        Args:
            patch_features: Features from the test image
            center_margin: Fraction of image to exclude from edges (0.2 = look at center 60%)
        """
        # Normalize patch features
        patch_features_norm = F.normalize(patch_features, dim=-1)

        # Compute similarity: for each test patch, find max similarity with any ref patch
        # Shape: [1, N_test, D] x [1, D, N_ref] -> [1, N_test, N_ref]
        all_similarities = torch.matmul(patch_features_norm, self.ref_features.transpose(-1, -2))

        # Take max over reference patches for each test patch
        # This finds test patches that match ANY part of the reference object
        similarity, _ = all_similarities.max(dim=-1)  # [1, N_test]
        similarity = similarity.squeeze(0)

        # Get grid dimensions
        num_patches = similarity.shape[0]
        grid_size = int(num_patches ** 0.5)

        # Reshape to grid and convert to numpy
        sim_grid = similarity.view(grid_size, grid_size).cpu().numpy()

        # Apply Gaussian blur to smooth the similarity map
        sim_blurred = cv2.GaussianBlur(sim_grid, (5, 5), 1.0)

        # Create binary mask using threshold
        mask = (sim_blurred > self.threshold).astype(np.uint8) * 255

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Opening: remove small noise (erode then dilate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Closing: fill small holes (dilate then erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Only look in center region - mask out edges
        margin = int(grid_size * center_margin)
        edge_mask = np.zeros_like(mask)
        edge_mask[margin:grid_size-margin, margin:grid_size-margin] = 255
        mask = cv2.bitwise_and(mask, edge_mask)

        # Find centroid of the mask, or fallback to max in blurred center
        if mask.sum() > 0:
            # Find contours and get largest one
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    max_x = int(M["m10"] / M["m00"])
                    max_y = int(M["m01"] / M["m00"])
                    max_val = sim_blurred[max_y, max_x]
                else:
                    # Fallback to max in mask
                    masked_sim = np.where(mask > 0, sim_blurred, 0)
                    max_idx = masked_sim.argmax()
                    max_y = max_idx // grid_size
                    max_x = max_idx % grid_size
                    max_val = sim_blurred[max_y, max_x]
            else:
                # No contours found, use max in center
                center_blurred = sim_blurred[margin:grid_size-margin, margin:grid_size-margin]
                max_val = center_blurred.max()
                max_idx = center_blurred.argmax()
                center_size = grid_size - 2 * margin
                max_y = max_idx // center_size + margin
                max_x = max_idx % center_size + margin
        else:
            # No mask above threshold, find max in center of blurred map
            center_blurred = sim_blurred[margin:grid_size-margin, margin:grid_size-margin]
            max_val = center_blurred.max()
            max_idx = center_blurred.argmax()
            center_size = grid_size - 2 * margin
            max_y = max_idx // center_size + margin
            max_x = max_idx % center_size + margin

        # Convert to normalized coordinates (0-1)
        norm_x = (max_x + 0.5) / grid_size
        norm_y = (max_y + 0.5) / grid_size

        return norm_x, norm_y, max_val, sim_blurred

    def image_callback(self, msg):
        """Process incoming camera image."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Get original dimensions
            orig_h, orig_w = cv_image.shape[:2]

            # Process FULL image for heatmap visualization
            pil_image = Image.fromarray(cv_image)
            img_tensor = self.pil_to_tensor(pil_image)
            patch_features = self.extract_patch_features(img_tensor)

            # Get full similarity grid for visualization
            patch_features_norm = F.normalize(patch_features, dim=-1)
            all_similarities = torch.matmul(patch_features_norm, self.ref_features.transpose(-1, -2))
            similarity, _ = all_similarities.max(dim=-1)
            similarity = similarity.squeeze(0)
            num_patches = similarity.shape[0]
            grid_size = int(num_patches ** 0.5)
            full_sim_grid = similarity.view(grid_size, grid_size).cpu().numpy()
            full_sim_grid = cv2.GaussianBlur(full_sim_grid, (5, 5), 1.0)

            # For tracking: only search in top half of the grid
            if self.search_top_half:
                search_grid = full_sim_grid[:grid_size // 2, :].copy()
                y_offset = 0  # Top half starts at 0
                search_h = orig_h // 2
            else:
                search_grid = full_sim_grid[grid_size // 2:, :].copy()
                y_offset = grid_size // 2
                search_h = orig_h // 2

            # Find best match in search region
            max_val = search_grid.max()
            max_idx = search_grid.argmax()
            search_grid_h, search_grid_w = search_grid.shape
            max_y_local = max_idx // search_grid_w
            max_x = max_idx % search_grid_w

            # Convert to full grid coordinates
            max_y = max_y_local + y_offset

            # Convert to pixel coordinates
            pixel_x = int((max_x + 0.5) / grid_size * orig_w)
            pixel_y = int((max_y + 0.5) / grid_size * orig_h)
            confidence = float(max_val)

            # Publish position if above threshold
            if confidence > self.threshold:
                point_msg = Point()
                point_msg.x = float(pixel_x)
                point_msg.y = float(pixel_y)
                point_msg.z = float(confidence)
                self.position_pub.publish(point_msg)

                self.get_logger().info(f"Tracked at ({pixel_x}, {pixel_y}), confidence: {confidence:.3f}")

            # Create debug visualization with FULL heatmap
            debug_img = cv_image.copy()

            # Draw horizontal line showing search boundary
            boundary_y = orig_h // 2
            cv2.line(debug_img, (0, boundary_y), (orig_w, boundary_y), (255, 255, 0), 2)

            # Draw FULL heatmap overlay
            sim_resized = cv2.resize(full_sim_grid, (orig_w, orig_h))
            sim_normalized = ((sim_resized - sim_resized.min()) / (sim_resized.max() - sim_resized.min() + 1e-8) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(sim_normalized, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            debug_img = cv2.addWeighted(debug_img, 0.6, heatmap, 0.4, 0)

            # Draw crosshair at best match
            if confidence > self.threshold:
                color = (0, 255, 0)  # Green if above threshold
            else:
                color = (255, 0, 0)  # Red if below

            cv2.circle(debug_img, (pixel_x, pixel_y), 10, color, 2)
            cv2.line(debug_img, (pixel_x - 20, pixel_y), (pixel_x + 20, pixel_y), color, 2)
            cv2.line(debug_img, (pixel_x, pixel_y - 20), (pixel_x, pixel_y + 20), color, 2)

            # Add confidence text
            cv2.putText(debug_img, f"Conf: {confidence:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='rgb8')
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DinoTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
