"""Test script for mask preprocessing mode using images from data/road.

This script tests the mask preprocessing functionality in run_policy_onnx.py
by processing images from data/road directory and visualizing the results.
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from run_policy_onnx import preprocess_image, _get_perspective_transform_matrix


class MaskPreprocessTester:
    """Test class for mask preprocessing mode."""
    
    def __init__(
        self,
        data_dir: str = "data/road",
        output_dir: str = "test_mask_outputs",
        model_width: int = 84,
        model_height: int = 84,
        cam_width: int = 320,
        cam_height: int = 240,
        mask_hsv_lower: tuple = (0, 100, 100),
        mask_hsv_upper: tuple = (180, 255, 255),
        max_images: int = 10,
    ):
        """Initialize the tester.
        
        Args:
            data_dir: Directory containing test images
            output_dir: Directory to save output visualizations
            model_width: Target width for model input
            model_height: Target height for model input
            cam_width: Camera image width (for perspective transform)
            cam_height: Camera image height (for perspective transform)
            mask_hsv_lower: HSV lower bound for mask extraction
            mask_hsv_upper: HSV upper bound for mask extraction
            max_images: Maximum number of images to process
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_width = model_width
        self.model_height = model_height
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.mask_hsv_lower = np.array(mask_hsv_lower, dtype=np.uint8)
        self.mask_hsv_upper = np.array(mask_hsv_upper, dtype=np.uint8)
        self.max_images = max_images
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get perspective transform matrix (not used for mask mode, but available)
        self.perspective_matrix, self.perspective_size = _get_perspective_transform_matrix(
            cam_width, cam_height
        )
    
    def load_images(self):
        """Load images from data directory."""
        image_files = sorted(glob.glob(str(self.data_dir / "*.jpg")))
        if not image_files:
            image_files = sorted(glob.glob(str(self.data_dir / "*.png")))
        
        if not image_files:
            raise ValueError(f"No images found in {self.data_dir}")
        
        # Limit number of images
        image_files = image_files[:self.max_images]
        print(f"Found {len(image_files)} images to process")
        return image_files
    
    def process_single_image(self, image_path: str):
        """Process a single image and return intermediate results.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing:
                - original: Original BGR image
                - rgb: RGB converted image
                - hsv: HSV converted image
                - mask: Binary mask
                - mask_resized: Resized mask
                - mask_3ch: 3-channel mask
                - final_output: Final preprocessed output (1CHW float32)
        """
        # Load image
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get original dimensions
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Resize to camera dimensions if needed
        if orig_w != self.cam_width or orig_h != self.cam_height:
            frame_bgr = cv2.resize(frame_bgr, (self.cam_width, self.cam_height), 
                                  interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB (as done in preprocess_image)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to HSV
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        # Extract mask using HSV thresholds
        mask = cv2.inRange(hsv, self.mask_hsv_lower, self.mask_hsv_upper)
        
        # Resize mask to target dimensions
        mask_resized = cv2.resize(mask, (self.model_width, self.model_height), 
                                  interpolation=cv2.INTER_AREA)
        
        # Stack single channel to 3 channels
        mask_3ch = np.stack([mask_resized, mask_resized, mask_resized], axis=2)
        
        # Get final output using preprocess_image function
        final_output = preprocess_image(
            frame_bgr,
            self.model_width,
            self.model_height,
            obs_mode="mask",
            perspective_matrix=None,
            perspective_size=None,
            mask_hsv_lower=self.mask_hsv_lower,
            mask_hsv_upper=self.mask_hsv_upper,
        )
        
        return {
            "original": frame_bgr,
            "rgb": frame_rgb,
            "hsv": hsv,
            "mask": mask,
            "mask_resized": mask_resized,
            "mask_3ch": mask_3ch,
            "final_output": final_output,
            "image_path": image_path,
        }
    
    def visualize_results(self, results: dict, save_path: str):
        """Create a visualization of the preprocessing pipeline.
        
        Args:
            results: Dictionary from process_single_image
            save_path: Path to save the visualization
        """
        original = results["original"]
        rgb = results["rgb"]
        hsv = results["hsv"]
        mask = results["mask"]
        mask_resized = results["mask_resized"]
        final_output = results["final_output"]
        
        # Convert final output back to HWC for visualization
        # final_output is (1, 3, H, W) float32 [0, 1]
        final_vis = final_output[0].transpose(1, 2, 0)  # (H, W, 3)
        final_vis = (final_vis * 255).astype(np.uint8)
        
        # Resize all images to same height for comparison
        target_h = 240
        scale = target_h / original.shape[0]
        target_w = int(original.shape[1] * scale)
        
        # Resize images for visualization
        orig_vis = cv2.resize(original, (target_w, target_h))
        rgb_vis = cv2.resize(rgb, (target_w, target_h))
        hsv_vis = cv2.resize(hsv, (target_w, target_h))
        mask_vis = cv2.resize(mask, (target_w, target_h))
        mask_resized_vis = cv2.resize(mask_resized, (self.model_width * 4, self.model_height * 4), 
                                      interpolation=cv2.INTER_NEAREST)
        final_vis_large = cv2.resize(final_vis, (self.model_width * 4, self.model_height * 4), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Convert HSV to BGR for visualization (HSV is hard to visualize directly)
        # Show H, S, V channels separately
        hsv_h = hsv[:, :, 0]
        hsv_s = hsv[:, :, 1]
        hsv_v = hsv[:, :, 2]
        hsv_h_vis = cv2.applyColorMap(cv2.resize(hsv_h, (target_w, target_h)), cv2.COLORMAP_HSV)
        hsv_s_vis = cv2.resize(cv2.cvtColor(hsv_s, cv2.COLOR_GRAY2BGR), (target_w, target_h))
        hsv_v_vis = cv2.resize(cv2.cvtColor(hsv_v, cv2.COLOR_GRAY2BGR), (target_w, target_h))
        
        # Convert mask to BGR for visualization
        mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        mask_resized_vis_bgr = cv2.cvtColor(mask_resized_vis, cv2.COLOR_GRAY2BGR)
        
        # Create visualization grid
        # Row 1: Original, RGB, HSV (H channel), HSV (S channel), HSV (V channel)
        row1 = np.hstack([orig_vis, rgb_vis, hsv_h_vis, hsv_s_vis, hsv_v_vis])
        
        # Row 2: Mask (full size), Mask (resized), Final output
        # Add labels
        def add_label(img, text):
            img_labeled = img.copy()
            cv2.putText(img_labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2, cv2.LINE_AA)
            return img_labeled
        
        mask_vis_labeled = add_label(mask_vis_bgr, "Mask (full)")
        mask_resized_vis_labeled = add_label(mask_resized_vis_bgr, "Mask (resized)")
        final_vis_labeled = add_label(final_vis_large, "Final output")
        
        # Ensure all images in row2 have the same height
        row2_height = target_h  # Use same height as row1
        mask_resized_vis_labeled = cv2.resize(mask_resized_vis_labeled, 
                                             (mask_resized_vis_labeled.shape[1], row2_height))
        final_vis_labeled = cv2.resize(final_vis_labeled, 
                                       (final_vis_labeled.shape[1], row2_height))
        
        # Pad to same width
        pad_w = max(mask_vis_labeled.shape[1], mask_resized_vis_labeled.shape[1], 
                   final_vis_labeled.shape[1])
        
        def pad_to_width(img, width):
            if img.shape[1] < width:
                pad = (width - img.shape[1]) // 2
                return cv2.copyMakeBorder(img, 0, 0, pad, width - img.shape[1] - pad, 
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return img
        
        row2 = np.hstack([
            pad_to_width(mask_vis_labeled, pad_w),
            pad_to_width(mask_resized_vis_labeled, pad_w),
            pad_to_width(final_vis_labeled, pad_w),
        ])
        
        # Ensure row1 and row2 have the same width
        row1_width = row1.shape[1]
        row2_width = row2.shape[1]
        if row1_width != row2_width:
            if row2_width < row1_width:
                # Pad row2 to match row1 width
                pad = (row1_width - row2_width) // 2
                row2 = cv2.copyMakeBorder(row2, 0, 0, pad, row1_width - row2_width - pad,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                # Resize row2 to match row1 width (maintain aspect ratio)
                scale = row1_width / row2_width
                new_h = int(row2.shape[0] * scale)
                row2 = cv2.resize(row2, (row1_width, new_h))
                # If height doesn't match, pad vertically
                if row2.shape[0] != row1.shape[0]:
                    pad = (row1.shape[0] - row2.shape[0]) // 2
                    row2 = cv2.copyMakeBorder(row2, pad, row1.shape[0] - row2.shape[0] - pad, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Combine rows
        vis = np.vstack([row1, row2])
        
        # Save visualization
        cv2.imwrite(save_path, vis)
        print(f"Saved visualization to: {save_path}")
    
    def run_tests(self):
        """Run tests on all images."""
        image_files = self.load_images()
        
        print(f"\nProcessing {len(image_files)} images...")
        print(f"HSV lower bound: {self.mask_hsv_lower}")
        print(f"HSV upper bound: {self.mask_hsv_upper}")
        print(f"Output directory: {self.output_dir}\n")
        
        for i, image_path in enumerate(image_files):
            try:
                print(f"Processing [{i+1}/{len(image_files)}]: {Path(image_path).name}")
                
                # Process image
                results = self.process_single_image(image_path)
                
                # Create visualization
                image_name = Path(image_path).stem
                save_path = self.output_dir / f"{image_name}_mask_vis.jpg"
                self.visualize_results(results, str(save_path))
                
                # Print statistics
                mask_pixels = np.sum(results["mask"] > 0)
                total_pixels = results["mask"].size
                mask_ratio = mask_pixels / total_pixels * 100
                print(f"  Mask coverage: {mask_ratio:.2f}% ({mask_pixels}/{total_pixels} pixels)")
                print(f"  Final output shape: {results['final_output'].shape}")
                print(f"  Final output dtype: {results['final_output'].dtype}")
                print(f"  Final output range: [{results['final_output'].min():.3f}, {results['final_output'].max():.3f}]")
                print()
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nTest completed! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test mask preprocessing mode")
    parser.add_argument("--data-dir", type=str, default="data/road",
                       help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="test_mask_outputs",
                       help="Directory to save output visualizations")
    parser.add_argument("--model-width", type=int, default=84,
                       help="Target width for model input")
    parser.add_argument("--model-height", type=int, default=84,
                       help="Target height for model input")
    parser.add_argument("--cam-width", type=int, default=320,
                       help="Camera image width")
    parser.add_argument("--cam-height", type=int, default=240,
                       help="Camera image height")
    parser.add_argument("--mask-hsv-lower", type=int, nargs=3, default=[0, 100, 100],
                       metavar=("H", "S", "V"),
                       help="HSV lower bound for mask extraction")
    parser.add_argument("--mask-hsv-upper", type=int, nargs=3, default=[180, 255, 255],
                       metavar=("H", "S", "V"),
                       help="HSV upper bound for mask extraction")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    tester = MaskPreprocessTester(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_width=args.model_width,
        model_height=args.model_height,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        mask_hsv_lower=tuple(args.mask_hsv_lower),
        mask_hsv_upper=tuple(args.mask_hsv_upper),
        max_images=args.max_images,
    )
    
    tester.run_tests()


if __name__ == "__main__":
    main()

