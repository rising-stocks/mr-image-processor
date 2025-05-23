#!/usr/bin/env python3
"""
Test Visualization Script for MR Image Processor

This script:
1. Downloads a sample DICOM file from a public dataset
2. Converts it to NIfTI format using the dicom_to_nifti.py script
3. Visualizes both the original DICOM and converted NIfTI images side by side
4. Cleans up temporary files if requested
"""

import os
import sys
import tempfile
import shutil
import subprocess
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import nibabel as nib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_dicom(target_dir):
    """
    Create a synthetic DICOM file with a test pattern for conversion testing.
    
    Args:
        target_dir: Directory where the synthetic DICOM file will be saved
        
    Returns:
        Path to the directory containing the synthetic DICOM file
    """
    logger.info("Creating synthetic DICOM file for testing")
    
    # Create a directory to store the DICOM file
    dicom_dir = os.path.join(target_dir, "synthetic_dicom")
    os.makedirs(dicom_dir, exist_ok=True)
    
    # Path to save the synthetic DICOM file
    dicom_file = os.path.join(dicom_dir, "synthetic.dcm")
    
    try:
        # Create a synthetic image with a recognizable pattern
        # We'll create a 256x256 image with a gradient and some geometric shapes
        pixel_array = np.zeros((256, 256), dtype=np.uint16)
        
        # Create a radial gradient
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        d = np.sqrt(x*x + y*y)
        gradient = (1 - d) * 2000  # Scale to a reasonable DICOM intensity range
        gradient = np.clip(gradient, 0, 4000).astype(np.uint16)
        
        # Add the gradient to the image
        pixel_array += gradient
        
        # Add a square in the center
        pixel_array[78:178, 78:178] = 3000
        
        # Add some circles
        for i in range(4):
            center_x, center_y = 64 + i*40, 64 + i*40
            for x in range(256):
                for y in range(256):
                    if (x - center_x)**2 + (y - center_y)**2 < 20**2:
                        pixel_array[y, x] = 4000
        
        # Create a new DICOM file
        # File meta info data elements
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # Create the FileDataset instance
        ds = FileDataset(dicom_file, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Add the data elements
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TEST12345"
        ds.Modality = "MR"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SecondaryCaptureDeviceManufacturer = "MR Image Processor"
        
        # Image related elements
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [1.0, 1.0]
        ds.Rows = pixel_array.shape[0]
        ds.Columns = pixel_array.shape[1]
        
        # Set creation date/time
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S.%f')
        
        # Set the pixel data
        ds.PixelData = pixel_array.tobytes()
        
        # Save the DICOM file
        ds.save_as(dicom_file)
        
        logger.info(f"Synthetic DICOM file created: {dicom_file}")
        return dicom_dir
        
    except Exception as e:
        logger.error(f"Error creating synthetic DICOM file: {e}")
        raise


def convert_dicom_to_nifti(dicom_dir, output_path, normalize=True):
    """
    Convert DICOM to NIfTI using the dicom_to_nifti.py script.
    
    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Path where the NIfTI file will be saved
        normalize: Whether to normalize the output to uint8
        
    Returns:
        Path to the converted NIfTI file
    """
    logger.info(f"Converting DICOM to NIfTI: {dicom_dir} -> {output_path}")
    
    # Build the command
    cmd = [sys.executable, "dicom_to_nifti.py", "--input", dicom_dir, "--output", output_path]
    
    if normalize:
        cmd.append("--normalize")
    
    # Run the conversion script
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Conversion output: {result.stdout}")
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Expected output file not found: {output_path}")
            
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting DICOM to NIfTI: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        raise


def visualize_dicom_and_nifti(dicom_dir, nifti_path):
    """
    Visualize the original DICOM and converted NIfTI images side by side.
    
    Args:
        dicom_dir: Directory containing DICOM files
        nifti_path: Path to the converted NIfTI file
    
    Returns:
        Path to the saved comparison image
    """
    logger.info("Visualizing DICOM and NIfTI images")
    
    try:
        # Load the NIfTI file
        nifti_img = nib.load(nifti_path)
        nifti_data = nifti_img.get_fdata()
        
        # For 3D images, use the middle slice
        if len(nifti_data.shape) == 3:
            middle_slice = nifti_data.shape[2] // 2
            nifti_slice = nifti_data[:, :, middle_slice]
        else:
            nifti_slice = nifti_data
        
        # Find a DICOM file in the directory
        dicom_file = os.path.join(dicom_dir, "synthetic.dcm")
        if not os.path.exists(dicom_file):
            # Try to find any DCM file if the expected one doesn't exist
            dicom_files = list(Path(dicom_dir).glob("*.dcm"))
            if not dicom_files:
                dicom_files = list(Path(dicom_dir).glob("*"))  # Try any file
                
            if not dicom_files:
                raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
            dicom_file = str(dicom_files[0])
            
        # Load the DICOM file
        dicom_data = pydicom.dcmread(dicom_file)
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the DICOM image
        ax1.imshow(dicom_data.pixel_array, cmap='gray')
        ax1.set_title('Original DICOM')
        ax1.axis('off')
        
        # Display the NIfTI image
        ax2.imshow(nifti_slice, cmap='gray')
        ax2.set_title('Converted NIfTI (Normalized)')
        ax2.axis('off')
        
        # Add overall title
        plt.suptitle('DICOM to NIfTI Conversion Comparison')
        plt.tight_layout()
        
        # Create docs/images directory if it doesn't exist
        docs_dir = os.path.join(os.getcwd(), "docs")
        images_dir = os.path.join(docs_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Always save the plot to the docs/images directory
        plot_file = os.path.join(images_dir, "dicom_nifti_comparison.png")
        plt.savefig(plot_file, dpi=150)
        logger.info(f"Comparison image saved to: {plot_file}")
        
        # Also show the plot interactively if a display is available
        if os.environ.get('DISPLAY') or plt.get_backend() != 'Agg':
            plt.show()
        
        return plot_file
        
    except Exception as e:
        logger.error(f"Error visualizing images: {e}")
        raise


def main():
    """Main function to test the DICOM to NIfTI conversion and visualization."""
    
    # Create a temporary directory for our test
    temp_dir = tempfile.mkdtemp()
    nifti_path = os.path.join(temp_dir, "output.nii.gz")
    
    # Set up non-interactive backend if running in environment without display
    if not os.environ.get('DISPLAY') and plt.get_backend() == 'TkAgg':
        logger.info("No display detected. Using non-interactive Agg backend")
        plt.switch_backend('Agg')
    
    try:
        # Step 1: Create synthetic DICOM
        dicom_dir = create_synthetic_dicom(temp_dir)
        
        # Step 2: Convert DICOM to NIfTI
        convert_dicom_to_nifti(dicom_dir, nifti_path, normalize=True)
        
        # Step 3: Visualize both images and save the comparison
        image_path = visualize_dicom_and_nifti(dicom_dir, nifti_path)
        logger.info(f"Comparison image available at: {image_path}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    finally:
        # Clean up temporary files
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

