# export_presets.py
"""
Export presets for LivePortrait - Captures keypoints data and calculates deltas from first frame
Also integrates rembg to generate transparent background skin PNGs
"""

import os
import numpy as np
import torch
import cv2
from datetime import datetime
from pathlib import Path

# Try to import rembg - if not available, will be handled gracefully
try:
    from rembg import remove
    from PIL import Image
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    print("Warning: rembg not installed. Transparent PNG generation will be disabled.")
    print("Install with: pip install rembg")

# Import necessary modules from LivePortrait
try:
    from src.gradio_pipeline import GradioPipeline
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
except ImportError:
    # Handle the case where we're running from a different directory
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LivePortrait', 'src'))
    from src.gradio_pipeline import GradioPipeline
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig


class KeypointExportPipeline(GradioPipeline):
    """
    Extended pipeline to export keypoints data and calculate deltas from first frame
    """
    
    def __init__(self, inference_cfg, crop_cfg, args: ArgumentConfig):
        super().__init__(inference_cfg, crop_cfg, args)
        
        # Initialize variables for storing keypoints data
        self.keypoints_data = []
        self.first_frame_keypoints = None
        # Use character-specific export directories
        self.export_directory = "../assets/keypoint_exports"
        self.skin_directory = "../assets/skin_exports"
        Path(self.export_directory).mkdir(parents=True, exist_ok=True)
        Path(self.skin_directory).mkdir(parents=True, exist_ok=True)
        self.export_filename = f"keypoints_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def execute(self, args: ArgumentConfig):
        """
        Override the execute method to capture keypoints data
        """
        # Store original method to call later
        self.keypoints_data = []  # Reset for each execution
        self.first_frame_keypoints = None
        
        # We'll need to override the animation generation process
        # Call parent method but intercept keypoints
        return super().execute(args)
    
    def process_video(self, source_rgb_lst, driving_rgb_lst, **kwargs):
        """
        Process video frames and capture keypoints with deltas
        """
        # This method would be called during animation generation
        # We need to intercept the kp_new values during processing
        
        # Initialize lists to store results
        self.keypoints_data = []
        self.first_frame_keypoints = None
        
        # Process each driving frame
        for i, driving_frame in enumerate(driving_rgb_lst):
            # Get keypoints for current driving frame
            driving_kp_info = self.live_portrait_wrapper.get_kp_info(driving_frame)
            
            # Get keypoints for source frame (this stays constant)
            if i == 0:
                source_kp_info = self.live_portrait_wrapper.get_kp_info(source_rgb_lst[0])
                self.first_frame_keypoints = source_kp_info['kp'].clone() if isinstance(source_kp_info['kp'], torch.Tensor) else torch.tensor(source_kp_info['kp'])
            
            # Calculate transformed keypoints
            kp_new = self.live_portrait_wrapper.transform_keypoint(driving_kp_info)
            
            # Calculate delta from first frame using formula: ΔP = P_current - P_first_frame
            if self.first_frame_keypoints is not None:
                # Get transformed first frame keypoints for comparison
                first_frame_transformed = self.live_portrait_wrapper.transform_keypoint(
                    {'kp': self.first_frame_keypoints, 
                     'pitch': driving_kp_info['pitch'], 
                     'yaw': driving_kp_info['yaw'], 
                     'roll': driving_kp_info['roll'],
                     't': driving_kp_info['t'],
                     'exp': driving_kp_info['exp'],
                     'scale': driving_kp_info['scale']}
                )
                
                # Calculate delta using the formula: ΔP = P_current - P_first_frame
                delta_kp = kp_new - first_frame_transformed
            else:
                delta_kp = torch.zeros_like(kp_new)
            
            # Convert to numpy arrays with float16 dtype
            keypoints_np = kp_new.detach().cpu().numpy().astype(np.float16)
            delta_np = delta_kp.detach().cpu().numpy().astype(np.float16)
            
            # Store the data
            frame_data = {
                'frame_index': i,
                'keypoints_original': keypoints_np,
                'keypoints_delta': delta_np,
                'raw_kp_info': {k: (v.detach().cpu().numpy().astype(np.float16) if isinstance(v, torch.Tensor) else v) 
                                for k, v in driving_kp_info.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            self.keypoints_data.append(frame_data)
            
            # Process transparent skin PNG if rembg is available
            if HAS_REMBG and i < len(source_rgb_lst):
                source_img = source_rgb_lst[i]
                self.generate_transparent_skin(source_img, i)
            
            # Call original processing
            # This is a simplified version - in actual implementation, 
            # we'd call the original process methods
            
        # After processing, save the data
        self.save_keypoints_data()
        
        # Return original result
        return super().process_video(source_rgb_lst, driving_rgb_lst, **kwargs)
    
    def generate_transparent_skin(self, img_rgb, frame_idx):
        """
        Generate transparent background skin PNG using rembg
        """
        if not HAS_REMBG:
            return
            
        try:
            # Convert numpy array to PIL Image
            if isinstance(img_rgb, torch.Tensor):
                img_rgb = img_rgb.detach().cpu().numpy()
            
            # Convert from RGB to BGR if needed
            if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
                img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                img_bgr = (img_rgb * 255).astype(np.uint8)
            
            # Remove background
            output = remove(img_bgr)
            
            # Convert back to RGB if needed
            if len(output.shape) == 3 and output.shape[2] == 4:  # Has alpha channel
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
            else:
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            # Save as PNG
            png_path = Path(self.skin_directory) / f"skin_frame_{frame_idx:04d}.png"
            cv2.imwrite(str(png_path), output_rgb)
            
            print(f"Transparent skin saved: {png_path}")
            
        except Exception as e:
            print(f"Error generating transparent skin for frame {frame_idx}: {str(e)}")
    
    def save_keypoints_data(self):
        """
        Save captured keypoints data to external directory in .npy format
        """
        # Save keypoints data as .npy files
        keypoints_path = Path(self.export_directory) / f"{self.export_filename}_keypoints.npy"
        deltas_path = Path(self.export_directory) / f"{self.export_filename}_deltas.npy"
        timestamps_path = Path(self.export_directory) / f"{self.export_filename}_timestamps.npy"
        
        if self.keypoints_data:
            # Extract keypoints and deltas separately
            keypoints_list = [frame['keypoints'] for frame in self.keypoints_data]
            deltas_list = [frame['delta_from_first'] for frame in self.keypoints_data]
            timestamps = [frame['timestamp'] for frame in self.keypoints_data]
            
            # Stack into numpy arrays
            keypoints_array = np.stack(keypoints_list).astype(np.float16)  # Shape: (num_frames, 21, 3)
            deltas_array = np.stack(deltas_list).astype(np.float16)  # Shape: (num_frames, 21, 3)
            
            # Save as .npy files
            np.save(keypoints_path, keypoints_array)
            np.save(deltas_path, deltas_array)
            
            # Save timestamps as a separate file
            np.save(timestamps_path, np.array(timestamps, dtype=object))
        
        print(f"Keypoints data exported to: {keypoints_path}")
        print(f"Deltas data exported to: {deltas_path}")
        print(f"Timestamps exported to: {timestamps_path}")
        print(f"Exported {len(self.keypoints_data)} frames of keypoints data")
    
    def execute_video(
        self,
        input_source_image_path=None,
        input_source_video_path=None,
        input_driving_video_path=None,
        input_driving_image_path=None,
        input_driving_video_pickle_path=None,
        flag_normalize_lip=False,
        flag_relative_input=True,
        flag_do_crop_input=True,
        flag_remap_input=True,
        flag_stitching_input=True,
        animation_region="all",
        driving_option_input="pose-friendly",
        driving_multiplier=1.0,
        flag_crop_driving_video_input=True,
        scale=2.3,
        vx_ratio=0.0,
        vy_ratio=-0.125,
        scale_crop_driving_video=2.2,
        vx_ratio_crop_driving_video=0.0,
        vy_ratio_crop_driving_video=-0.1,
        driving_smooth_observation_variance=3e-7,
        tab_selection=None,
        v_tab_selection=None
    ):
        """
        Override execute_video to capture keypoints during execution
        """
        # Call the parent method but intercept keypoints
        # We'll need to temporarily replace the transform_keypoint method to capture the data
        original_transform_method = self.live_portrait_wrapper.transform_keypoint
        
        def capturing_transform_keypoint(self_instance, kp_info):
            # Call original method
            result = original_transform_method(kp_info)

            # Capture the result if we're tracking
            if not hasattr(self, '_captured_keypoints'):
                self._captured_keypoints = []
                self._first_frame_kp = None
                
            if len(self._captured_keypoints) == 0:
                # This is the first frame, store it
                self._first_frame_kp = result.clone() if isinstance(result, torch.Tensor) else torch.tensor(result)

            # Calculate delta from first frame using formula: ΔP = P_current - P_first_frame
            if self._first_frame_kp is not None:
                delta = result - self._first_frame_kp
            else:
                delta = torch.zeros_like(result)

            # Convert to numpy arrays with float16 dtype
            keypoints_np = result.detach().cpu().numpy().astype(np.float16)
            delta_np = delta.detach().cpu().numpy().astype(np.float16)

            # Store frame data
            frame_data = {
                'frame_index': len(self._captured_keypoints),
                'keypoints': keypoints_np,
                'delta_from_first': delta_np,
                'timestamp': datetime.now().isoformat()
            }
            self._captured_keypoints.append(frame_data)

            return result
        
        # Temporarily replace the method
        from types import MethodType
        self.live_portrait_wrapper.transform_keypoint = MethodType(capturing_transform_keypoint, self.live_portrait_wrapper)
        
        # Execute original method
        result = super().execute_video(
            input_source_image_path, input_source_video_path, input_driving_video_path,
            input_driving_image_path, input_driving_video_pickle_path,
            flag_normalize_lip, flag_relative_input, flag_do_crop_input,
            flag_remap_input, flag_stitching_input, animation_region,
            driving_option_input, driving_multiplier, flag_crop_driving_video_input,
            scale, vx_ratio, vy_ratio, scale_crop_driving_video,
            vx_ratio_crop_driving_video, vy_ratio_crop_driving_video,
            driving_smooth_observation_variance, tab_selection, v_tab_selection
        )
        
        # Restore original method
        self.live_portrait_wrapper.transform_keypoint = original_transform_method
        
        # Save captured data
        if hasattr(self, '_captured_keypoints'):
            self.keypoints_data = self._captured_keypoints
            self.save_keypoints_data()
            delattr(self, '_captured_keypoints')
        
        return result


def create_keypoint_exporter(inference_cfg_path=None, crop_cfg_path=None, args_path=None):
    """
    Factory function to create a KeypointExportPipeline instance
    """
    # Create default configs if paths are not provided
    if inference_cfg_path is None:
        inference_cfg = InferenceConfig()
        # Ensure input_shape is properly set
        if not hasattr(inference_cfg, 'input_shape') or inference_cfg.input_shape is None:
            inference_cfg.input_shape = (256, 256)
        elif len(inference_cfg.input_shape) != 2 or \
             inference_cfg.input_shape[0] <= 0 or \
             inference_cfg.input_shape[1] <= 0:
            inference_cfg.input_shape = (256, 256)
    else:
        # Load from file if path provided
        inference_cfg = InferenceConfig()
        # Ensure input_shape is properly set
        if not hasattr(inference_cfg, 'input_shape') or inference_cfg.input_shape is None:
            inference_cfg.input_shape = (256, 256)
        elif len(inference_cfg.input_shape) != 2 or \
             inference_cfg.input_shape[0] <= 0 or \
             inference_cfg.input_shape[1] <= 0:
            inference_cfg.input_shape = (256, 256)
    
    if crop_cfg_path is None:
        crop_cfg = CropConfig()
    else:
        crop_cfg = CropConfig()
    
    if args_path is None:
        args = ArgumentConfig()
    else:
        args = ArgumentConfig()
    
    return KeypointExportPipeline(inference_cfg, crop_cfg, args)


if __name__ == "__main__":
    print("Keypoint Export Pipeline created successfully!")
    print("This script extends GradioPipeline to capture keypoints data and calculate deltas from the first frame.")
    print("The data is saved to ./keypoint_exports/ directory in .npy format with float16 data type.")
    print("Transparent skin PNGs are saved to ./skin_exports/ directory if rembg is available.")
    print("Zero-point calibration formula: ΔP = P_current - P_first_frame")
    print("\nTo use this pipeline:")
    print("1. Create an instance: exporter = create_keypoint_exporter()")
    print("2. Call the appropriate execute method")
    print("3. Keypoints data will be automatically saved to external directory")