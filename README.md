# MR Image Processor

A Python tool for converting MR DICOM files to NIfTI format with optional normalization to uint8 (0-255) range.

## Overview

This tool provides a simple and efficient way to:
1. Convert medical MR DICOM series to the NIfTI format commonly used in research
2. Normalize the image intensity to the uint8 range (0-255), which is useful for visualization and certain processing algorithms
3. Maintain proper metadata during the conversion process

The script is designed to be used from the command line with straightforward arguments, making it suitable for batch processing and integration into larger workflows.

## Features

- DICOM series to NIfTI conversion with SimpleITK backend
- Optional normalization to uint8 (0-255) range
- Preservation of important image metadata (spacing, origin, etc.)
- Comprehensive error handling and validation
- Detailed logging
- Command-line interface with intuitive arguments

## Requirements

- Python 3.6+
- SimpleITK
- nibabel
- numpy

## Installation

### Using pip (recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/rising-stocks/mr-image-processor.git
   cd mr-image-processor
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Manual Installation

If you prefer to install the dependencies manually:

```bash
pip install SimpleITK nibabel numpy
```

## Usage

### Basic Usage

Convert a DICOM series to NIfTI format:

```bash
python dicom_to_nifti.py --input /path/to/dicom/folder --output output.nii.gz
```

### With Normalization

Convert and normalize to uint8 (0-255) range:

```bash
python dicom_to_nifti.py --input /path/to/dicom/folder --output output.nii.gz --normalize
```

### Verbose Mode

Enable detailed logging:

```bash
python dicom_to_nifti.py --input /path/to/dicom/folder --output output.nii.gz --verbose
```

### Command Line Arguments

- `-i, --input`: Input directory containing DICOM files (required)
- `-o, --output`: Output NIfTI file path (required)
- `-n, --normalize`: Normalize output to uint8 (0-255) range
- `-v, --verbose`: Enable verbose logging

## Error Handling

The script includes comprehensive error handling for common issues:

- Input directory validation (existence, permissions)
- DICOM series validation
- Output file path validation
- Runtime exceptions

When an error occurs, the script will:
1. Log a detailed error message
2. Return a non-zero exit code
3. Provide suggestions for troubleshooting when possible

## Limitations

- The tool is designed for MR DICOM series and may not work optimally for other modalities
- Multi-series DICOM directories are not fully supported (will use first series)
- The normalization is a linear scaling to 0-255 range, which may not be optimal for all use cases
- Complex or non-standard DICOM formats may not convert correctly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

