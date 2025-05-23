#!/usr/bin/env python3
"""
DICOM to NIfTI Converter with Normalization

This script converts MR DICOM series to NIfTI format and optionally normalizes
the output to uint8 (0-255) range. It provides command-line arguments for
specifying input and output paths, as well as normalization options.

Usage:
    python dicom_to_nifti.py --input /path/to/dicom/folder --output /path/to/output.nii.gz --normalize
"""

import os
import sys
import argparse
import logging
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dicom_to_nifti")


def validate_input_directory(directory_path):
    """
    Validate that the input directory exists and contains DICOM files.
    
    Args:
        directory_path: Path to the directory containing DICOM files
        
    Returns:
        bool: True if directory is valid, raises exception otherwise
    """
    path = Path(directory_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory_path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory_path}")
    
    # Check for DICOM files (simple check for .dcm extension)
    dicom_files = list(path.glob("*.dcm"))
    if not dicom_files:
        # Try without extension since some DICOM files don't have .dcm extension
        try:
            reader = sitk.ImageSeriesReader()
            series_IDs = reader.GetGDCMSeriesIDs(str(path))
            if not series_IDs:
                raise ValueError(f"No DICOM series found in directory: {directory_path}")
        except Exception as e:
            raise ValueError(f"Error reading DICOM directory: {e}")
    
    return True


def validate_output_path(output_path):
    """
    Validate that the output directory exists and is writable.
    
    Args:
        output_path: Path where the output NIfTI file will be saved
        
    Returns:
        bool: True if output path is valid, raises exception otherwise
    """
    path = Path(output_path)
    
    # Check if the parent directory exists and is writable
    parent_dir = path.parent
    if not parent_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {parent_dir}")
    
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"No write permission for output directory: {parent_dir}")
    
    # Check file extension
    if path.suffix not in ['.nii', '.gz'] and not str(path).endswith('.nii.gz'):
        logger.warning(f"Output filename doesn't have standard NIfTI extension (.nii or .nii.gz): {output_path}")
    
    return True


def read_dicom_series(dicom_dir):
    """
    Read a DICOM series from the specified directory.
    
    Args:
        dicom_dir: Path to directory containing DICOM files
        
    Returns:
        SimpleITK.Image: The loaded image
    """
    logger.info(f"Reading DICOM series from: {dicom_dir}")
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    if not dicom_names:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")
    
    reader.SetFileNames(dicom_names)
    try:
        image = reader.Execute()
        logger.info(f"Successfully read DICOM series with size: {image.GetSize()}")
        return image
    except Exception as e:
        raise RuntimeError(f"Error reading DICOM series: {e}")


def normalize_to_uint8(image_data):
    """
    Normalize image data to uint8 format (0-255).
    
    Args:
        image_data: numpy array containing image data
        
    Returns:
        numpy.ndarray: Normalized image data as uint8
    """
    logger.info("Normalizing image to uint8 format")
    
    # Get min and max values
    data_min = np.min(image_data)
    data_max = np.max(image_data)
    
    # Handle case where min=max (constant image)
    if data_min == data_max:
        logger.warning("Image has constant value. Normalization will result in all zeros.")
        return np.zeros_like(image_data, dtype=np.uint8)
    
    # Normalize to 0-255 range
    normalized_data = ((image_data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    
    logger.info(f"Normalization complete. Original range: [{data_min}, {data_max}], New range: [0, 255]")
    return normalized_data


def convert_dicom_to_nifti(dicom_dir, output_path, normalize=False):
    """
    Convert DICOM series to NIfTI format and optionally normalize to uint8.
    
    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Path where NIfTI file will be saved
        normalize: Whether to normalize the image to uint8 (0-255) range
        
    Returns:
        Path: Path to the saved NIfTI file
    """
    # Read DICOM series
    image = read_dicom_series(dicom_dir)
    
    # Convert SimpleITK image to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Get original orientation information
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    
    # Normalize to uint8 if requested
    if normalize:
        array = normalize_to_uint8(array)
        logger.info(f"Image normalized to uint8. Shape: {array.shape}, dtype: {array.dtype}")
    
    # Convert to NIfTI format using nibabel
    # SimpleITK and nibabel have different axis conventions, so we need to reorient
    # DICOM data from SimpleITK is in [z, y, x] format, we need to convert to [x, y, z]
    array = np.transpose(array, (2, 1, 0))
    
    # Create NIfTI image
    affine = np.eye(4)
    # Set spacing in the affine matrix
    for i in range(3):
        affine[i, i] = spacing[i]
    # Set origin in the affine matrix
    affine[:3, 3] = origin
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(array, affine)
    
    # Save the NIfTI file
    output_path = Path(output_path)
    nib.save(nifti_img, output_path)
    
    logger.info(f"NIfTI image saved to: {output_path}")
    
    return output_path


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to NIfTI format with optional normalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Input directory containing DICOM files"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Output NIfTI file path (e.g., output.nii.gz)"
    )
    
    parser.add_argument(
        "-n", "--normalize", 
        action="store_true", 
        help="Normalize output to uint8 (0-255) range"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function to execute the DICOM to NIfTI conversion."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set logging level based on verbose flag
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
        
        # Log the arguments
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Normalization: {'enabled' if args.normalize else 'disabled'}")
        
        # Validate input and output paths
        validate_input_directory(args.input)
        validate_output_path(args.output)
        
        # Perform conversion
        convert_dicom_to_nifti(args.input, args.output, args.normalize)
        
        logger.info("Conversion completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

