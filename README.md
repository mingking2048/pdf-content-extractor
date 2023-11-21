# README for PDF Content Extractor

## Overview

The PDF Content Extractor is designed to efficiently extract text, tables, and figures from PDF files. It leverages `pdfplumber` for text and table extraction and integrates with an image analysis module for processing figures. This tool is particularly useful for academic and research purposes where quick extraction of content from PDF documents is required.

## Prerequisites

Before using the PDF Content Extractor, ensure the following prerequisites are met:

1. **Python Version**: Requires Python 3.8 or higher.
2. **Libraries**: Dependencies include pdfplumber, torch, PIL, and other dependencies listed in `requirements.txt`.
3. **Environment**: A Conda environment is recommended for managing dependencies.

## Installation

Follow these steps to install the tool:

1. **Create a Conda Environment**:
   ```bash
   conda create -n pdf_extractor python=3.8 -y
   conda activate pdf_extractor
   ```

2. **Clone the Repository**:
   - Clone the repository from GitHub and navigate to the project directory.

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

## Usage

### Basic Usage

1. **Prepare Your PDFs**:
   - Place the PDF files you want to analyze in a designated directory.

2. To analyze images, run the script with the necessary arguments:
   ```bash
   python pdf_image_analysis.py --input_path <path-to-pdf> --prompt <custom-prompt> --keyword <keyword>
   ```
   
3. To analyze textual data, figures and tables, run the `pdf_ext.py` script:
   - Use the following command to run the script:
	```bash
		python pdf_ext.py \
		--input_path <path-to-pdf> \
		--output_path <path-to-output> \
		--prompt <custom-prompt> \
		--keyword <keyword> \
		--convert_table --convert_figure
    ```
   - The `--convert_table` flag is used to extract tables, and `--convert_figure` is for figure analysis.
   - Note: keyword will be ignored when prompt is used

### Advanced Usage

- Customize prompts to guide the response generation for extracted images and textual content.
- Choose different configs for LLaVA models for varied response styles and complexities.

## Features

- **Text Extraction**: Extracts plain text from PDF documents.
- **Table Extraction**: Efficiently identifies and extracts tables.
- **Figure Analysis**: Integrates with an image analysis module to analyze figures within the document.
- **Custom Output**: Outputs extracted content to a specified file.

## Example

For image analysis:
```bash
python pdf_image_analysis.py --input_path "pdf/starGAN.pdf" --keyword "GAN"
```

For pdf text extraction:
```bash
python pdf_ext.py --input_path "pdf/starGAN.pdf" --output_path "output.txt" --keyword "GAN" --convert_table --convert_figure
```

## Dependencies

The tool requires:

- **pdfplumber**: For extracting text and tables from PDFs.
- **Image Analysis Module**: For analyzing figures (additional setup may be required).

## Contributing

Interested in contributing? [Provide guidelines for contributions, including how to submit pull requests, coding standards, and issue reporting guidelines.]

## Acknowledgements

- [PDFPlumber](https://github.com/jsvine/pdfplumber): The primary library used for PDF text and table extraction.
- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main): the codebase we built upon.

---
