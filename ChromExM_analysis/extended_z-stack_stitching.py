#!/usr/bin/env python3

import os
import glob
import csv
import argparse
import datetime
import numpy as np
import tifffile as tiff
import re

from aicsimageio import AICSImage


"""
Z-stack stitching for sequentially acquired 3D microscopy volumes.

This script stitches pairs of partially overlapping 3D image stacks acquired sequentially from the same field of view. It identifies the Z-overlap between the end of the first stack and the beginning of the second stack, then appends only the non-overlapping portion of the second stack.

The overlap is estimated by comparing the final slice of the first stack against candidate slices from the beginning of the second stack. Optional global XY registration can be applied to correct small rigid shifts between stacks.

Input organization
------------------
- The input directory contains one subfolder per acquired stack.
- Each subfolder contains one OME-TIFF, or one logical image series split across multiple TIFF files.
- Stack pairs are inferred from subfolder names using the pattern "..._E<group>_<index>".
- For example, "..._40x_E1_1" and "..._40x_E1_2" are stitched together within group E1.

Outputs
-------
For each stitched pair, the script writes:
- A stitched TIFF image
- stitch_report.csv
- Diagnostic images in a _debug folder

Metadata
--------
Physical voxel sizes are read with AICSImageIO and propagated to ImageJ-compatible TIFF outputs.
"""

# ----------------------------
# Global cache 
# ----------------------------
metadata_cache = {}

# ----------------------------
# Simple logging helper
# ----------------------------
def log(message, log_file=None):
    """
    Log a message to stdout and optionally append it to a log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =====================================================================
# Main functions
# =====================================================================

def universal_read_tiff(file_path, log_file=None):
    """
    Universal function to read an OME-TIFF file, extract metadata, and load all channels.

    Args:
        file_path (str): Path to the OME-TIFF file.
        log_file (str, optional): Path to a log file where messages are recorded.

    Returns:
        tuple: A tuple containing the AICSImage object, the image stack (np.ndarray), and metadata.
    """
    log(f"Reading file: {file_path}", log_file)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found or invalid: {file_path}")
    try:
        # Read the file using AICSImageIO
        img = AICSImage(file_path)

        # Load the full image stack including all channels
        # Shape order: (C, Z, Y, X)
        image_stack = img.get_image_data("CZYX")
        metadata = {
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "channel_names": img.channel_names,
        }
        print(metadata)
        log(f"Loaded image stack shape: {image_stack.shape} (C, Z, Y, X)", log_file)

        return img, image_stack, metadata
    except Exception as e:
        log(f"Error reading file {file_path}: {e}", log_file)
        raise

def universal_read_tiff_multi(file_paths, log_file=None):
    """
    Universal function to read multiple OME-TIFF files as a single series.

    Args:
        file_paths (list of str): List of paths to TIFF files in the same subfolder.
        log_file (str, optional): Path to a log file where messages are recorded.

    Returns:
        tuple: A tuple containing the combined image stack (np.ndarray) and metadata (dict).
    """
    log(f"Reading files as a single multi-series image: {file_paths}", log_file)

    try:
        # Use the first file to represent the series (AICSImage can infer the full set)
        representative_file = file_paths[0]
        img = AICSImage(representative_file)

        # Read the entire series as one logical stack
        stack = img.get_image_data("CZYX")  # Shape: (C, Z, Y, X)
        metadata = {
            "physical_pixel_sizes": img.physical_pixel_sizes,
            "channel_names": img.channel_names,
        }

        # Log the loaded stack details
        log(f"Loaded combined image stack shape: {stack.shape} (C, Z, Y, X)", log_file)

        return stack, metadata

    except Exception as e:
        log(f"Error reading files as a single multi-series image: {e}", log_file)
        raise

def combined_reading(input_dir, log_file=None):
    """
    Read TIFF files from subfolders and determine whether to use single or multi-file reading logic.

    Args:
        input_dir (str): Path to the directory containing subfolders with TIFF files.
        log_file (str, optional): Path to a log file for recording progress and errors.

    Yields:
        dict: Contains the subfolder name, combined stack, and metadata.
              Format: {"subfolder": subfolder, "stack": combined_stack, "metadata": metadata}
    """
    # Find all subfolders in the input directory
    subfolders = [
        subfolder for subfolder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, subfolder))
    ]

    # Iterate over subfolders to read TIFF files
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_dir, subfolder)
        tiff_files = sorted(set(
            glob.glob(os.path.join(subfolder_path, "*.ome.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tiff"))
        ))

        if not tiff_files:
            log(f"No OME-TIFF files found in subfolder: {subfolder_path}", log_file)
            continue

        try:
            # Read and combine multiple TIFF files if needed
            if len(tiff_files) > 1:
                log(f"Multiple TIFF files found in {subfolder_path}. Combining files.", log_file)
                combined_stack, metadata = universal_read_tiff_multi(tiff_files, log_file)
            else:
                # Single file case
                img, combined_stack, metadata = universal_read_tiff(tiff_files[0], log_file)

            # Yield the results for this subfolder
            yield {
                "subfolder": subfolder,
                "stack": combined_stack,
                "metadata": metadata,
            }

        except Exception as e:
            log(f"Error processing subfolder {subfolder}: {e}", log_file)

def parse_group_and_index(folder_name: str):
    """
    Extract sample prefix, group, and index from names ending in:
      ..._E1_1
      ..._E1_2

    Returns:
      (sample_key, group_str, index_int) or (None, None, None)
    """
    m = re.search(r"^(.*)_((?:E|e)\d+)_([0-9]+)$", folder_name)
    if not m:
        return None, None, None

    sample_key = m.group(1)
    group = m.group(2).upper()
    idx = int(m.group(3))
    return sample_key, group, idx
    
def combined_reading_pairs_streaming(input_dir, log_file=None):
    """
    Read and yield pairs of image stacks one pair at a time.

    The input directory is scanned for stack subfolders, which are sorted and
    processed as sequential pairs. Only the current pair is loaded into memory,
    reducing memory usage for large datasets.

    Yields:
        dict containing:
            {
                "A_subfolder": str,
                "B_subfolder": str,
                "A_stack": np.ndarray,
                "B_stack": np.ndarray,
                "A_metadata": dict,
                "B_metadata": dict,
            }

    Notes:
        - Reads single- or multi-file TIFF datasets automatically.
        - Skips subfolders containing no TIFF files.
        - If an odd number of readable subfolders is present, the final
          unpaired subfolder is skipped.
    """

    def _read_one_subfolder(subfolder):
        """
        Read TIFF files from a single subfolder using the same logic as combined_reading().
        Returns (stack, metadata) or (None, None) if unreadable.
        """
        subfolder_path = os.path.join(input_dir, subfolder)

        tiff_files = sorted(set(
            glob.glob(os.path.join(subfolder_path, "*.ome.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tiff"))
        ))

        if not tiff_files:
            log(f"No OME-TIFF files found in subfolder: {subfolder_path}", log_file)
            return None, None

        try:
            if len(tiff_files) > 1:
                log(f"Multiple TIFF files found in {subfolder_path}. Combining files.", log_file)
                stack, metadata = universal_read_tiff_multi(tiff_files, log_file)
            else:
                img, stack, metadata = universal_read_tiff(tiff_files[0], log_file)

            return stack, metadata

        except Exception as e:
            log(f"Error processing subfolder {subfolder}: {e}", log_file)
            return None, None

   # Find all subfolders in the input directory (names only)
    subfolders = [
        subfolder for subfolder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, subfolder))
    ]

    # Build groups { "E1": [(idx, name), ...], "E2": [...], ... }
    groups = {}

    for name in subfolders:
        subfolder_path = os.path.join(input_dir, name)

        # lightweight existence check: do not read image data here
        tiff_files = (
            glob.glob(os.path.join(subfolder_path, "*.ome.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tif")) +
            glob.glob(os.path.join(subfolder_path, "*.tiff"))
        )
        if not tiff_files:
            log(f"No OME-TIFF files found in subfolder: {subfolder_path}", log_file)
            continue

        sample_key, g, idx = parse_group_and_index(name)
        if g is None:
            log(f"Skipping subfolder with unexpected name format (no E#_index at end): {name}", log_file)
            continue

        group_key = (sample_key, g)
        groups.setdefault(group_key, []).append((idx, name))

    if not groups:
        log("No valid E#_index subfolders found to stitch.", log_file)
        return

    # Sort groups by E number (E1, E2, E3, ...)
    group_names_sorted = sorted(
        groups.keys(),
        key=lambda x: (x[0], int(x[1][1:]) if x[1][1:].isdigit() else x[1])
    )

    # Count total pairs and sort entries within each group by idx
    total_pairs = 0
    for g in group_names_sorted:
        groups[g].sort(key=lambda t: t[0])  # sort by idx
        total_pairs += len(groups[g]) // 2

    log(
        f"Found {sum(len(v) for v in groups.values())} readable stacks across {len(groups)} group(s) -> {total_pairs} pair(s).",
        log_file
    )

    pair_counter = 0
    for g in group_names_sorted:
        entries = groups[g]  # list of (idx, name) sorted

        if len(entries) < 2:
            continue

        if len(entries) % 2 != 0:
            log(f"Group {g} has odd count ({len(entries)}). The last one will be skipped in this group.", log_file)

        # Pair sequentially within the same group: (1,2), (3,4), ...
        for j in range(0, len(entries) - 1, 2):
            idx_a, a_sub = entries[j]
            idx_b, b_sub = entries[j + 1]

            pair_counter += 1
            log(f"Reading pair {pair_counter}/{total_pairs} in {g}: A={a_sub}, B={b_sub}", log_file)

            a_stack, a_meta = _read_one_subfolder(a_sub)
            b_stack, b_meta = _read_one_subfolder(b_sub)

            if a_stack is None or b_stack is None:
                log(f"Skipping pair A={a_sub}, B={b_sub} because one or both stacks failed to read.", log_file)
                continue

            yield {
                "A_subfolder": a_sub,
                "B_subfolder": b_sub,
                "A_stack": a_stack,
                "B_stack": b_stack,
                "A_metadata": a_meta,
                "B_metadata": b_meta,
            }



def save_as_ome_tiff(output_stack, output_file, metadata=None):
    """
    Save a modified NumPy array as an OME-TIFF file with metadata, ensuring correct axis interpretation.

    Args:
        output_stack (np.ndarray): The modified image data to save.
        output_file (str): Path to save the new OME-TIFF file.
        metadata (dict): Metadata dictionary containing relevant information.
    """
    # Persistent variable to store the last metadata used
    if not hasattr(save_as_ome_tiff, "_last_metadata"):
        save_as_ome_tiff._last_metadata = None  # Initialize if not already set

    # Use provided metadata, or fall back to the last metadata if available
    if metadata is not None:
        save_as_ome_tiff._last_metadata = metadata
    elif save_as_ome_tiff._last_metadata is None:
        raise ValueError("Metadata must be provided for the first call.")

    current_metadata = save_as_ome_tiff._last_metadata    

    try:
        # Extract voxel sizes (physical pixel sizes)
        physical_pixel_sizes = current_metadata.get("physical_pixel_sizes", (1.0, 1.0, 1.0))
        pixel_size_x = physical_pixel_sizes[1]  # X spacing
        pixel_size_y = physical_pixel_sizes[2]  # Y spacing
        pixel_size_z = physical_pixel_sizes[0]  # Z spacing

        # Calculate resolution in dots per micrometer for X and Y
        resolution_x = 1 / pixel_size_x  # In microns
        resolution_y = 1 / pixel_size_y  # In microns

        # Handle dimensionality
        if output_stack.ndim == 4:  # Multi-channel stack (C, Z, Y, X)
            # Reorder to (Z, C, Y, X) for ImageJ compatibility
            output_stack = np.transpose(output_stack, (1, 0, 2, 3))
            # Define LUTs for multi-channel images
            true_colors = [
                [0, 255, 255],  # Cyan for channel 1
                [255, 0, 255],  # Magenta for channel 2
                [255, 255, 0],  # Yellow for channel 3
                [128, 128, 128],  # Gray for channel 4
            ]
            num_channels = output_stack.shape[1]
            luts = []
            for i in range(num_channels):
                r, g, b = true_colors[i % len(true_colors)]
                lut = np.array([
                    np.linspace(0, r, 256),  # Red gradient
                    np.linspace(0, g, 256),  # Green gradient
                    np.linspace(0, b, 256),  # Blue gradient
                ], dtype=np.uint8)
                luts.append(lut)
            # ImageJ-specific metadata for multi-channel images
            imagej_metadata = {
                "spacing": pixel_size_z,  # Z voxel size
                "unit": "micron",         # Unit for all dimensions
                "mode": "composite",      # Open in composite mode
                "LUTs": luts,             # Assign LUTs to channels
            }

        elif output_stack.ndim == 3:  # Single-channel stack (Z, Y, X)
            output_stack = np.transpose(output_stack, (0, 1, 2))
            imagej_metadata = {
                "spacing": pixel_size_z,
                "unit": "micron",
                "channels": 0,  # Explicitly state that the image has one channel
                "slices": output_stack.shape[0],  # Number of Z slices
                "frames": 1,  # Default to 1 frame for simplicity
                "axes": "ZYX",
            }
           
        else:
            raise ValueError(f"Unexpected output stack dimensions: {output_stack.ndim}")
        # Write the TIFF file with proper ImageJ metadata
        tiff.imwrite(
            output_file,
            output_stack,
            photometric="minisblack",
            resolution=(resolution_x, resolution_y),  # Resolution for X and Y
            metadata=imagej_metadata,                # Include Z-spacing, composite mode, and LUTs
            imagej=True                              # Ensure compatibility with ImageJ
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save OME-TIFF file {output_file}: {e}")



def split_channels(image_stack, metadata, channels, output_dir, subfolder, log_file):
    """
    Split channels from a multichannel stack and save each channel with voxel calibration.

    Args:
        image_stack (np.ndarray): Multichannel image stack to process.
        metadata (dict): Metadata containing voxel size information.
        channels (dict): Dictionary mapping channel names to indices (1-based).
        output_dir (str): Path to save split channel images.
        subfolder (str): Subfolder name used to construct output filenames.
        log_file (str): Path to a log file where the process is logged.
    """
    global metadata_cache  # Access the global metadata cache

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if metadata is already cached for this subfolder
    if subfolder in metadata_cache:
        log(f"Using cached metadata for subfolder: {subfolder}", log_file)
        cached_metadata = metadata_cache[subfolder]
    else:
        log(f"Caching metadata for subfolder: {subfolder}", log_file)
        metadata_cache[subfolder] = metadata
        cached_metadata = metadata

    for channel_name, channel_index in channels.items():
        # Create a subfolder for each channel
        channel_dir = os.path.join(output_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        # Construct the output path
        output_path = os.path.join(channel_dir, f"C{channel_index}-{subfolder}.tif")

        # Validate output path
        if not isinstance(output_path, str):
            raise ValueError(f"Invalid output path: {output_path}")

        # Skip if output already exists
        if os.path.exists(output_path):
            log(f"Skipping split for {channel_name} in {subfolder} (output exists).", log_file)
            continue

        # Extract and save the specified channel
        try:
            channel_image = image_stack[channel_index - 1]
            save_as_ome_tiff(channel_image, output_path, cached_metadata)
            log(f"Saved {channel_name} to {output_path} with voxel calibration.", log_file)
        except Exception as e:
            log(f"Error saving {channel_name} for {subfolder}: {e}", log_file)


# =====================================================================
# Stitching parameters
# =====================================================================

# Minimum and maximum number of overlapping Z slices to evaluate.
K_MIN = 3
K_MAX = 100

# Minimum normalized cross-correlation score required to accept an overlap.
# Lower-scoring matches are treated as failed overlap detection.
MIN_ACCEPTABLE_SCORE = 0.25

# If multiple overlap lengths have nearly identical scores, select the
# smallest overlap within this tolerance of the best score.
NEAR_TIE_TOL = 1e-3

# Constant intensity offset subtracted before overlap matching.
# This value may need adjustment for different microscopes or acquisition settings.
BG_OFFSET = 104.0

def _ensure_czyx(stack):
    """
    Ensure stack is in CZYX form.
    - If 3D (Z,Y,X): treat as single-channel and add C dimension.
    - If 4D: assume already CZYX.
    """
    if stack.ndim == 3:
        return stack[np.newaxis, ...]
    if stack.ndim == 4:
        return stack
    raise ValueError(f"Unexpected stack ndim={stack.ndim}. Expected 3 or 4.")


def _normalized_cross_correlation(a, b):
    """
    Compute NCC between two same-shaped arrays.
    Returns a scalar in [-1, 1] (or -inf if undefined).
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    a_mean = float(a.mean())
    b_mean = float(b.mean())

    a0 = a - a_mean
    b0 = b - b_mean

    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("-inf")

    return float(np.sum(a0 * b0) / denom)


    
def _phase_corr_shift(a2d, b2d):
    """
    Estimate integer-pixel (dy, dx) shift to align b2d to a2d using phase correlation.
    Returns (dy, dx). Positive dy means b should be shifted DOWN to match a.
    """
    a = a2d.astype(np.float32, copy=False)
    b = b2d.astype(np.float32, copy=False)

    a -= float(a.mean())
    b -= float(b.mean())

    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R) + 1e-12
    R /= denom

    cc = np.fft.ifft2(R)
    cc = np.abs(cc)

    peak = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = peak

    # wrap-around to signed shift
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]

    return int(dy), int(dx)

def _shift2d(img, dy, dx):
    """Integer shift with zero fill (no wrap)."""
    out = np.zeros_like(img)
    y0_src = max(0, -dy)
    y1_src = min(img.shape[0], img.shape[0] - dy)
    x0_src = max(0, -dx)
    x1_src = min(img.shape[1], img.shape[1] - dx)

    y0_dst = max(0, dy)
    y1_dst = min(img.shape[0], img.shape[0] + dy)
    x0_dst = max(0, dx)
    x1_dst = min(img.shape[1], img.shape[1] + dx)

    out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out

def find_overlap_k_seam_xyreg_with_dup_penalty(
    stack_a,
    stack_b,
    channel_for_matching=0,
    bg_offset=104.0,
    search_limit_b=220,
    downsample=4,
    near_tie_tol=0.002,
    min_score=0.15,
):
    """
    Estimate the Z-overlap between two sequential image stacks.

    The final slice of stack A is matched against candidate slices near the
    beginning of stack B after optional global XY registration. If the best
    match is B[j], the inferred overlap is j + 1 and stitching appends B[j+1:].

    Returns:
        tuple: best_k, best_score, second_best_score, status, reason,
        best_dy, best_dx, ks, scores_list, shifts
    """
    a = _ensure_czyx(stack_a)
    b = _ensure_czyx(stack_b)

    if a.shape[0] <= channel_for_matching or b.shape[0] <= channel_for_matching:
        return None, None, None, "skip", "Channel index out of range.", 0, 0, [], [], []

    a_vol = a[channel_for_matching].astype(np.float32, copy=False)  # ZYX
    b_vol = b[channel_for_matching].astype(np.float32, copy=False)  # ZYX

    z_a = a_vol.shape[0]
    z_b = b_vol.shape[0]
    if z_a < 1 or z_b < 1:
        return None, None, None, "skip", "Stacks too short.", 0, 0, [], [], []

    # BG subtract for matching only
    a_bs = a_vol - bg_offset
    b_bs = b_vol - bg_offset
    np.maximum(a_bs, 0.0, out=a_bs)
    np.maximum(b_bs, 0.0, out=b_bs)

    ds = int(max(1, downsample))
    A_last = a_bs[-1, ::ds, ::ds]
    if not np.any(A_last > 0):
        return None, None, None, "skip", "A last slice empty after BG subtraction.", 0, 0, [], [], []

    def gradmag2d(img):
        gy, gx = np.gradient(img.astype(np.float32, copy=False))
        return np.sqrt(gx * gx + gy * gy, dtype=np.float32)

    A_ref = gradmag2d(A_last)

    max_j = min(int(search_limit_b), z_b)
    ks = []
    scores = []
    shifts = []

    for j in range(max_j):
        # score candidate using B[j] (the slice that matches A[-1])
        B_score = gradmag2d(b_bs[j, ::ds, ::ds])
        dy_score, dx_score = _phase_corr_shift(A_ref, B_score)
        score = _normalized_cross_correlation(A_ref, _shift2d(B_score, dy_score, dx_score))

        #store the shift for the true boundary slice B[j+1] (the first slice we keep)
        if j + 1 < z_b:
            B_boundary = gradmag2d(b_bs[j + 1, ::ds, ::ds])
            dy_use, dx_use = _phase_corr_shift(A_ref, B_boundary)
        else:
            dy_use, dx_use = dy_score, dx_score

        ks.append(int(j + 1))               # overlap_k
        scores.append(float(score))         # score for overlap selection
        shifts.append((int(dy_use), int(dx_use)))  # shift to APPLY

    scores_arr = np.array(scores, dtype=np.float64)
    if not np.any(np.isfinite(scores_arr)):
        return None, None, None, "skip", "All candidate scores non-finite.", 0, 0, ks, scores, shifts

    best_score_raw = float(np.nanmax(scores_arr))
    tie_idx = np.where(scores_arr >= (best_score_raw - float(near_tie_tol)))[0]
    if tie_idx.size == 0:
        return None, None, None, "skip", "Failed to identify best overlap candidate.", 0, 0, ks, scores, shifts

    # Prefer larger k among near-ties to reduce duplication
    best_i = int(tie_idx[-1])

    best_k = int(ks[best_i])
    best_score = float(scores_arr[best_i])
    best_dy, best_dx = shifts[best_i]

    # second-best (excluding the chosen best_i)
    tmp = scores_arr.copy()
    tmp[best_i] = -np.inf
    second_best = float(np.nanmax(tmp)) if np.any(np.isfinite(tmp)) else float("nan")

    if not np.isfinite(best_score) or best_score < float(min_score):
        return best_k, best_score, second_best, "skip", f"Low-confidence score={best_score:.4f}", best_dy, best_dx, ks, scores, shifts

    return best_k, best_score, second_best, "ok", "Overlap found (last-slice anchored).", best_dy, best_dx, ks, scores, shifts

def stitch_pair_zonly_keep_a_append_b(stack_a, stack_b, overlap_k):
    """
    Create stitched stack by keeping A fully and appending B[overlap_k:] along Z, for all channels.
    Expects stacks in CZYX or ZYX (ZYX will be treated as single-channel).
    """
    a = _ensure_czyx(stack_a)
    b = _ensure_czyx(stack_b)

    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Channel count mismatch: A has {a.shape[0]} channels, B has {b.shape[0]} channels.")

    if overlap_k is None:
        raise ValueError("overlap_k cannot be None.")

    if overlap_k < 0 or overlap_k > b.shape[1]:
        raise ValueError(f"Invalid overlap_k={overlap_k} for B with z={b.shape[1]}.")

    # Concatenate along Z axis (axis=1 in CZYX)
    stitched = np.concatenate([a, b[:, overlap_k:, :, :]], axis=1)
    return stitched


def write_stitch_report_csv(report_path, row_dict):
    """
    Write a single-row stitch_report.csv (overwrite if exists).
    """
    fieldnames = list(row_dict.keys())
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row_dict)

def _shift3d_zyx(vol_zyx, dy, dx):
    """Shift a ZYX volume in XY by (dy, dx) with zero fill."""
    out = np.zeros_like(vol_zyx)
    z = vol_zyx.shape[0]
    for i in range(z):
        out[i] = _shift2d(vol_zyx[i], dy, dx)
    return out

def shift_stack_czyx_xy(stack_czyx, dy, dx):
    """Shift an entire CZYX stack in XY by (dy, dx) with zero fill."""
    s = _ensure_czyx(stack_czyx)
    out = np.zeros_like(s)
    for c in range(s.shape[0]):
        out[c] = _shift3d_zyx(s[c], dy, dx)
    return out

def debug_dump_pair(
    out_dir,
    a_stack,
    b_stack,
    channel=0,
    bg_offset=104.0,
    seam_w=15,
    downsample=4,
    ks=None,
    scores=None,
    shifts=None,
    best_k=None,
    best_dy=0,
    best_dx=0,
    note=""
):
    os.makedirs(out_dir, exist_ok=True)

    a = _ensure_czyx(a_stack)[channel].astype(np.float32, copy=False)  # ZYX
    b = _ensure_czyx(b_stack)[channel].astype(np.float32, copy=False)  # ZYX

    # BG subtract for visualization 
    a_bs = a - bg_offset
    b_bs = b - bg_offset
    np.maximum(a_bs, 0.0, out=a_bs)
    np.maximum(b_bs, 0.0, out=b_bs)

    ds = int(max(1, downsample))

    # Save scores table if provided
    if ks is not None and scores is not None:
        csv_path = os.path.join(out_dir, "scores.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["overlap_k", "score", "dy", "dx"])
            for i, k in enumerate(ks):
                dy = shifts[i][0] if shifts is not None else ""
                dx = shifts[i][1] if shifts is not None else ""
                w.writerow([k, float(scores[i]), dy, dx])

    # Make seam reference images for the chosen best_k
    if best_k is not None and best_k >= seam_w + 1 and best_k <= b_bs.shape[0]:
        A_seam = a_bs[-seam_w:, :, :].mean(axis=0)[::ds, ::ds]
        B_seam = b_bs[best_k - seam_w:best_k, :, :].mean(axis=0)[::ds, ::ds]

        B_aligned = _shift2d(B_seam, best_dy, best_dx)
        diff = np.abs(A_seam - B_aligned)

        def norm_u8(img):
            v = img.copy()
            v -= float(np.min(v))
            m = float(np.max(v))
            if m > 0:
                v /= m
            v = (v * 255.0).astype(np.uint8)
            return v

        tiff.imwrite(os.path.join(out_dir, "A_seam_mean.tif"), A_seam.astype(np.float32))
        tiff.imwrite(os.path.join(out_dir, "B_seam_mean.tif"), B_seam.astype(np.float32))
        tiff.imwrite(os.path.join(out_dir, "B_seam_aligned.tif"), B_aligned.astype(np.float32))
        tiff.imwrite(os.path.join(out_dir, "seam_absdiff.tif"), diff.astype(np.float32))

        # also write PNGs for quick check
        try:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(out_dir, "A_seam_mean.png"), norm_u8(A_seam))
            imageio.imwrite(os.path.join(out_dir, "B_seam_mean.png"), norm_u8(B_seam))
            imageio.imwrite(os.path.join(out_dir, "B_seam_aligned.png"), norm_u8(B_aligned))
            imageio.imwrite(os.path.join(out_dir, "seam_absdiff.png"), norm_u8(diff))
        except Exception:
            pass

    # Save boundary-slice diagnostics
    if best_k is not None and 0 <= (best_k - 1) < b_bs.shape[0]:
        A_boundary = a_bs[-1, :, :][::ds, ::ds]

        # First retained slice from B is index best_k because best_k = j + 1.
        if best_k < b_bs.shape[0]:
            B_boundary = b_bs[best_k, :, :][::ds, ::ds]
            B_boundary_aligned = _shift2d(B_boundary, best_dy, best_dx)
            diff_boundary = np.abs(A_boundary - B_boundary_aligned)

            tiff.imwrite(os.path.join(out_dir, "A_boundary.tif"), A_boundary.astype(np.float32))
            tiff.imwrite(os.path.join(out_dir, "B_boundary.tif"), B_boundary.astype(np.float32))
            tiff.imwrite(os.path.join(out_dir, "B_boundary_aligned.tif"), B_boundary_aligned.astype(np.float32))
            tiff.imwrite(os.path.join(out_dir, "boundary_absdiff.tif"), diff_boundary.astype(np.float32))

    # Save maximum-intensity projections from the end of A and beginning of B.
    a_tail = a_bs[max(0, a_bs.shape[0] - 60):, :, :]
    b_head = b_bs[:min(200, b_bs.shape[0]), :, :]

    A_mip = a_tail.max(axis=0)[::ds, ::ds]
    B_mip = b_head.max(axis=0)[::ds, ::ds]

    try:
        import imageio.v2 as imageio
        strip = np.concatenate([A_mip, B_mip], axis=1)
        strip_u8 = strip.copy()
        strip_u8 -= float(strip_u8.min())
        m = float(strip_u8.max())
        if m > 0:
            strip_u8 /= m
        strip_u8 = (strip_u8 * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, "A_tail_MIP__B_head_MIP.png"), strip_u8)
    except Exception:
        pass

    # save diagnostic note
    with open(os.path.join(out_dir, "debug_note.txt"), "w", encoding="utf-8") as f:
        f.write(note + "\n")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Z-only stitching of sequential A/B stack subfolders. Keeps A fully and appends non-overlapping portion of B."
    )
    parser.add_argument("--input", required=True, help="Input directory containing subfolders (each subfolder contains TIFF(s) for one stack).")
    parser.add_argument("--output", required=False, default=None,
                        help="Optional output directory. Default: <parent_of_input>/stitched")
    parser.add_argument("--log", required=False, default=None,
                        help="Optional log file path. Default: <output>/stitching.log")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input is not a directory: {input_dir}")

    # Default output: stitched/ in the parent dir of input
    if args.output is None:
        parent_dir = os.path.dirname(input_dir.rstrip(os.sep))
        output_root = os.path.join(parent_dir, "stitched")
    else:
        output_root = os.path.abspath(args.output)

    os.makedirs(output_root, exist_ok=True)

    log_file = args.log
    if log_file is None:
        log_file = os.path.join(output_root, "stitching.log")

    log(f"Input directory: {input_dir}", log_file)
    log(f"Output root: {output_root}", log_file)

    pair_index = 0
    for pair in combined_reading_pairs_streaming(input_dir, log_file=log_file):
        pair_index += 1

        a_name = pair["A_subfolder"]
        b_name = pair["B_subfolder"]

        stack_a = pair["A_stack"]
        stack_b = pair["B_stack"]
        meta_a  = pair["A_metadata"]

        log(f"Stitching pair {pair_index}: A={a_name}, B={b_name}", log_file)

        out_subfolder_name = f"stitched_{a_name}"
        out_dir = os.path.join(output_root, out_subfolder_name)
        os.makedirs(out_dir, exist_ok=True)

        report_path = os.path.join(out_dir, "stitch_report.csv")
        out_tif_path = os.path.join(out_dir, f"{out_subfolder_name}.tif")  

        # --- if output exists, skip  ---
        if os.path.exists(out_tif_path):
            log(f"Output exists, skipping: {out_tif_path}", log_file)
            write_stitch_report_csv(report_path, {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "stack_A": a_name,
                "stack_B": b_name,
                "status": "skip",
                "reason": "Output already exists.",
                "overlap_k": "",
                "best_score": "",
                "second_best_score": "",
                "z_A": _ensure_czyx(stack_a).shape[1],
                "z_B": _ensure_czyx(stack_b).shape[1],
                "z_out": "",
                "output_tif": out_tif_path,
            })
            continue

        # --- run overlap finder ---
        best_k, best_score, second_best_score, status, reason, best_dy, best_dx, ks, scores_list, shifts = find_overlap_k_seam_xyreg_with_dup_penalty(
            stack_a,
            stack_b,
            channel_for_matching=0,
            bg_offset=104.0,
            search_limit_b=220,
            downsample=4,
            min_score=0.15,
        )

        debug_dump_pair(
            out_dir=os.path.join(out_dir, "_debug"),
            a_stack=stack_a,
            b_stack=stack_b,
            channel=0,
            bg_offset=104.0,
            seam_w=7,
            downsample=4,
            ks=ks,
            scores=scores_list,
            shifts=shifts,
            best_k=best_k,
            best_dy=best_dy,
            best_dx=best_dx,
            note=f"A={a_name}\nB={b_name}\nstatus={status}\nreason={reason}\nbest_k={best_k}\nbest_score={best_score}\ndy={best_dy} dx={best_dx}\n"
        )

        z_a = _ensure_czyx(stack_a).shape[1]
        z_b = _ensure_czyx(stack_b).shape[1]

        if status != "ok":
            log(f"Skipping pair A={a_name}, B={b_name}: {reason}", log_file)
            write_stitch_report_csv(report_path, {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "stack_A": a_name,
                "stack_B": b_name,
                "status": "skip",
                "reason": reason,
                "overlap_k": best_k if best_k is not None else "",
                "best_score": f"{best_score:.6f}" if best_score is not None and np.isfinite(best_score) else "",
                "second_best_score": f"{second_best_score:.6f}" if second_best_score is not None and np.isfinite(second_best_score) else "",
                "z_A": z_a,
                "z_B": z_b,
                "z_out": "",
                "xy_dy": best_dy,
                "xy_dx": best_dx,
                "output_tif": out_tif_path,
            })
            continue

        try:
            # Shift entire B stack in XY using the seam-estimated shift
            # best_dy/best_dx were estimated on downsampled images (step=downsample),
            # so they are rescaled up to full-resolution before shifting the full-res stack.
            ds = 4  # must match the downsample from the overlap finder
            dy_full = int(best_dy * ds)
            dx_full = int(best_dx * ds)

            stack_b_aligned = shift_stack_czyx_xy(stack_b, dy_full, dx_full)

            # Now stitch using the aligned B
            stitched = stitch_pair_zonly_keep_a_append_b(stack_a, stack_b_aligned, best_k)
            z_out = _ensure_czyx(stitched).shape[1]

            save_as_ome_tiff(stitched, out_tif_path, meta_a)

            log(f"Saved stitched TIFF: {out_tif_path} (overlap_k={best_k}, score={best_score:.4f})", log_file)

            write_stitch_report_csv(report_path, {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "stack_A": a_name,
                "stack_B": b_name,
                "status": "ok",
                "reason": "Stitched successfully.",
                "overlap_k": best_k,
                "best_score": f"{best_score:.6f}",
                "second_best_score": f"{second_best_score:.6f}" if second_best_score is not None and np.isfinite(second_best_score) else "",
                "z_A": z_a,
                "z_B": z_b,
                "z_out": z_out,
                "output_tif": out_tif_path,
            })

        except Exception as e:
            msg = f"Error stitching/saving pair A={a_name}, B={b_name}: {e}"
            log(msg, log_file)
            write_stitch_report_csv(report_path, {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "stack_A": a_name,
                "stack_B": b_name,
                "status": "skip",
                "reason": msg,
                "overlap_k": best_k if best_k is not None else "",
                "best_score": f"{best_score:.6f}" if best_score is not None and np.isfinite(best_score) else "",
                "second_best_score": f"{second_best_score:.6f}" if second_best_score is not None and np.isfinite(second_best_score) else "",
                "z_A": z_a,
                "z_B": z_b,
                "z_out": "",
                "output_tif": out_tif_path,
            })
            continue

    log("Done.", log_file)


if __name__ == "__main__":
    main()