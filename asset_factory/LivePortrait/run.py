# run.py
"""
LivePortrait Character-Based Automation Pipeline

This script implements a character-based automation pipeline that:
1. Creates character-specific directory structures
2. Processes all driving videos from a fixed source folder
3. Generates character-specific manifests
4. Maintains float16 precision and hardcoded flags
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import torch
import json
from datetime import datetime

# Add LivePortrait src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from export_presets import create_keypoint_exporter
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.io import load_image_rgb, load_video
from src.utils.helper import basename


def check_neutral_face(video_path, first_frame_threshold=0.1):
    """
    Check if the first frame of the driving video is neutral face.
    This function performs basic validation to detect if the first frame
    has excessive expression that might indicate non-neutral pose.
    
    Args:
        video_path: Path to the driving video
        first_frame_threshold: Threshold for detecting non-neutral face
    
    Returns:
        True if neutral face detected, False otherwise
    """
    # Initialize LivePortraitWrapper to get keypoints
    inference_cfg = InferenceConfig()
    
    # Explicitly set the input shape to ensure it's valid
    if not hasattr(inference_cfg, 'input_shape') or inference_cfg.input_shape is None:
        print("Warning: input_shape not found in config, using default (256, 256)")
        inference_cfg.input_shape = (256, 256)
    elif len(inference_cfg.input_shape) != 2 or \
         inference_cfg.input_shape[0] <= 0 or \
         inference_cfg.input_shape[1] <= 0:
        print(f"Warning: Invalid input_shape {inference_cfg.input_shape}, using default (256, 256)")
        inference_cfg.input_shape = (256, 256)
    
    wrapper = LivePortraitWrapper(inference_cfg)
    
    # Load first frame of video
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read first frame of {video_path}")
        return False
    
    # Convert BGR to RGB
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Verify frame dimensions
    if len(first_frame_rgb.shape) != 3:
        print(f"Error: Expected 3D frame, got shape {first_frame_rgb.shape}")
        return False
    
    h, w = first_frame_rgb.shape[:2]
    target_h, target_w = inference_cfg.input_shape
    
    print(f"Frame dimensions: {w}x{h}, Target: {target_w}x{target_h}")
    
    # Verify that target dimensions are valid before attempting resize
    if target_w <= 0 or target_h <= 0:
        print(f"Error: Invalid target dimensions: {target_w}x{target_h}")
        return False
    
    # Prepare frame for processing
    # Note: prepare_source expects a 3D image (H, W, 3), not (1, H, W, 3)
    # Remove the [None, ...] operation which adds an extra dimension
    try:
        prepared_frame = wrapper.prepare_source(first_frame_rgb)
    except Exception as e:
        print(f"Error preparing source frame: {str(e)}")
        # Print additional debug info
        print(f"Frame shape: {first_frame_rgb.shape}")
        print(f"Input shape config: {inference_cfg.input_shape}")
        print(f"Target resize dimensions: {inference_cfg.input_shape[0]}, {inference_cfg.input_shape[1]}")
        return False
    
    # Get keypoints for first frame
    kp_info = wrapper.get_kp_info(prepared_frame)
    
    # Check if expression values are within neutral range
    # Expression values in neutral face should be relatively small
    exp_values = kp_info['exp']
    if isinstance(exp_values, torch.Tensor):
        exp_values = exp_values.detach().cpu().numpy()
    
    # Calculate magnitude of expression values
    exp_magnitude = np.linalg.norm(exp_values)
    
    # If expression magnitude is too high, it's likely not neutral
    if exp_magnitude > first_frame_threshold:
        print(f"Warning: First frame of {video_path} has high expression magnitude ({exp_magnitude:.4f}). "
              f"This may not be a neutral face. Threshold: {first_frame_threshold}")
        return False
    
    print(f"First frame of {video_path} appears to be neutral (magnitude: {exp_magnitude:.4f})")
    return True


def generate_character_manifest(character_dir):
    """
    Generate character-specific manifest.json in the character directory
    """
    motions_dir = character_dir / "motions"
    
    # Scan motions directory for .npy files
    motion_files = list(motions_dir.glob("*_deltas.npy"))
    
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "character": character_dir.name,
        "motions": []
    }
    
    for motion_file in motion_files:
        # Load the .npy file to get frame count
        try:
            motion_data = np.load(str(motion_file))
            frame_count = motion_data.shape[0] if len(motion_data.shape) > 0 else 0
            
            # Extract motion name from filename
            motion_name = motion_file.stem.replace("_deltas", "")
            
            manifest["motions"].append({
                "name": motion_name,
                "frames": int(frame_count),  # Convert to Python int for JSON serialization
                "file": f"motions/{motion_file.name}",
                "keypoints_file": f"motions/{motion_file.name.replace('_deltas', '_keypoints')}",
                "timestamps_file": f"motions/{motion_file.name.replace('_deltas', '_timestamps')}"
            })
            
        except Exception as e:
            print(f"Error processing motion file {motion_file}: {str(e)}")
    
    # Write manifest.json in the character directory
    manifest_path = character_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Character manifest generated at: {manifest_path}")
    print(f"Total motions in manifest: {len(manifest['motions'])}")


def process_single_video_for_character(source_path, driving_path, character_dir, neutral_check=True):
    """
    Process a single driving video for a specific character
    
    Args:
        source_path: Path to the golden source image
        driving_path: Path to the driving video
        character_dir: Path to the character directory
        neutral_check: Whether to check if first frame is neutral
    """
    # Validate inputs
    if not Path(source_path).exists():
        print(f"Error: Source file does not exist: {source_path}")
        return False
    
    if not Path(driving_path).exists():
        print(f"Error: Driving file does not exist: {driving_path}")
        return False
    
    # Perform neutral face check if enabled
    if neutral_check and not check_neutral_face(driving_path):
        print(f"Neutral face check failed for {driving_path}. Skipping this video.")
        return False
    
    print(f"Processing: {driving_path}")
    
    # Create default configurations
    inference_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    arg_cfg = ArgumentConfig()
    
    # Set paths
    arg_cfg.source = source_path
    arg_cfg.driving = driving_path
    
    # Set output to character directory
    arg_cfg.output_dir = str(character_dir)
    
    # Hardcoded parameters - these should NOT be changed
    arg_cfg.flag_relative_motion = True      # DO NOT ALLOW USER TO OVERRIDE
    arg_cfg.flag_do_crop = True              # DO NOT ALLOW USER TO OVERRIDE
    arg_cfg.flag_pasteback = True            # DO NOT ALLOW USER TO OVERRIDE
    arg_cfg.flag_stitching = True            # DO NOT ALLOW USER TO OVERRIDE
    
    # Create the export pipeline
    exporter = create_keypoint_exporter(inference_cfg, crop_cfg, arg_cfg)
    
    try:
        # Execute the pipeline
        result = exporter.execute_video(
            input_source_image_path=source_path,
            input_driving_video_path=driving_path,
            flag_relative_input=True,         # HARDCODED
            flag_do_crop_input=True,          # HARDCODED
            flag_remap_input=True,            # HARDCODED
            flag_stitching_input=True,        # HARDCODED
            animation_region="all",
            tab_selection='Image',
            v_tab_selection='Video'
        )
        
        # Get input name from driving video
        input_name = Path(driving_path).stem  # Gets name without extension
        export_dir = Path(exporter.export_directory)
        
        # Define character-specific directories
        motions_dir = character_dir / "motions"
        motions_dir.mkdir(exist_ok=True)
        
        # Find the generated files and move them to motions dir with renaming
        generated_files = list(export_dir.glob(f"*keypoints.npy"))
        if generated_files:
            # Get the most recently created file
            keypoints_file = max(generated_files, key=os.path.getctime)
            new_keypoints_name = motions_dir / f"{input_name}_keypoints.npy"
            # Convert to float16 before saving
            keypoints_data = np.load(str(keypoints_file))
            np.save(str(new_keypoints_name), keypoints_data.astype(np.float16))
            keypoints_file.unlink()  # Remove original file
            print(f"Moved keypoints file to: {new_keypoints_name}")
        
        generated_deltas = list(export_dir.glob(f"*deltas.npy"))
        if generated_deltas:
            # Get the most recently created file
            deltas_file = max(generated_deltas, key=os.path.getctime)
            new_deltas_name = motions_dir / f"{input_name}_deltas.npy"
            # Convert to float16 before saving
            deltas_data = np.load(str(deltas_file))
            np.save(str(new_deltas_name), deltas_data.astype(np.float16))
            deltas_file.unlink()  # Remove original file
            print(f"Moved deltas file to: {new_deltas_name}")
        
        generated_timestamps = list(export_dir.glob(f"*timestamps.npy"))
        if generated_timestamps:
            # Get the most recently created file
            timestamps_file = max(generated_timestamps, key=os.path.getctime)
            new_timestamps_name = motions_dir / f"{input_name}_timestamps.npy"
            # Convert to float16 before saving (though timestamps might be objects)
            timestamps_data = np.load(str(timestamps_file), allow_pickle=True)
            np.save(str(new_timestamps_name), timestamps_data)
            timestamps_file.unlink()  # Remove original file
            print(f"Moved timestamps file to: {new_timestamps_name}")
        
        # Move PNG files to character root directory as skin.png
        skin_dir = Path(exporter.skin_directory)
        if skin_dir.exists():
            # Look for PNG files in the skin directory
            png_files = list(skin_dir.glob("*.png"))
            if png_files:
                # Take the first PNG file and rename it to skin.png in the character root
                original_png = png_files[0]
                skin_path = character_dir / "skin.png"
                original_png.rename(skin_path)
                print(f"Renamed and moved skin PNG to: {skin_path}")
            
            # Clean up empty skin directory
            if skin_dir.exists() and not any(skin_dir.iterdir()):
                skin_dir.rmdir()
        
        print(f"Completed processing: {driving_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {driving_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def process_character_pipeline(source_path, character_name, neutral_check=True):
    """
    Main pipeline for processing a character: creates directory structure,
    processes all driving videos from fixed source, and generates manifest
    
    Args:
        source_path: Path to the golden source image
        character_name: Name of the character
        neutral_check: Whether to check if first frame is neutral
    """
    # Define the fixed driving source folder
    # Using driving videos from wife_assets
    driving_source_dir = Path("./wife_assets/driving_source/")
    
    if not driving_source_dir.exists():
        print(f"Error: Driving source directory does not exist: {driving_source_dir}")
        print("Please create the directory and add driving videos to it.")
        return False
    
    # Create character directory structure
    character_root = Path("../assets/avatars") / character_name
    character_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Created character directory: {character_root}")
    
    # Find all driving videos in the fixed source directory
    driving_videos = list(driving_source_dir.glob("*.mp4"))
    
    if not driving_videos:
        print(f"No MP4 files found in {driving_source_dir}")
        return False
    
    print(f"Found {len(driving_videos)} driving videos to process for character {character_name}")
    
    # Process each driving video for the character
    success_count = 0
    for driving_video in driving_videos:
        print(f"\n--- Processing {driving_video.name} for {character_name} ---")
        if process_single_video_for_character(
            source_path=source_path,
            driving_path=str(driving_video),
            character_dir=character_root,
            neutral_check=neutral_check
        ):
            success_count += 1
        else:
            print(f"Failed to process {driving_video.name}")
    
    # Generate character-specific manifest
    generate_character_manifest(character_root)
    
    print(f"\nCharacter pipeline completed: {success_count}/{len(driving_videos)} videos processed successfully for {character_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description='LivePortrait Character-Based Automation Pipeline')
    
    # Required arguments
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Path to the golden standard source image (the reference portrait)')
    parser.add_argument('--character', '-c', type=str, required=True,
                        help='Name of the character (e.g., Murasame_01)')
    
    # Optional arguments
    parser.add_argument('--no-neutral-check', action='store_true',
                        help='Skip neutral face check for first frame of driving videos')
    
    args = parser.parse_args()
    
    # Process the character pipeline
    print(f"Starting character pipeline for: {args.character}")
    print(f"Using source image: {args.source}")
    
    success = process_character_pipeline(
        source_path=args.source,
        character_name=args.character,
        neutral_check=not args.no_neutral_check
    )
    
    if success:
        print(f"\nPipeline completed successfully for character: {args.character}")
    else:
        print(f"\nPipeline failed for character: {args.character}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 80)
    print("LIVEPORTRAIT CHARACTER-BASED AUTOMATION PIPELINE")
    print("=" * 80)
    print(__doc__)
    print("=" * 80)
    
    main()