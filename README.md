# README for LLaVA PDF Image Analysis and Response Generation Tool

## Overview

LLaVA (Language Llama Variational Autoencoder) is a powerful tool designed to extract images from PDF files, particularly from academic papers, and generate descriptive responses using advanced language and visual processing models. It can analyze images in scientific papers, identify and extract them along with their captions or nearby text, and use a pre-trained LLaVA model to generate responses based on custom prompts.

## Prerequisites

Ensure the following prerequisites are met before using this tool:

1. **Python Version**: Python 3.8 or higher.
2. **Libraries**: PyMuPDF, pdfplumber, torch, PIL, and other dependencies listed in `pyproject.toml`.
3. **Conda Environment**: Recommended for managing dependencies.

## Installation

1. Create a Conda environment:
   ```bash
   conda create -n llava python=3.8 -y
   conda activate llava
   ```

2. Clone the repository and navigate to the project directory.

3. Install the project in editable mode and its dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

## Usage

### Basic Usage

1. Place your PDF files in a designated directory.
2. Run the script with the necessary arguments:
   ```bash
   python pdf_image_analysis.py --paper_path <path-to-pdf> --prompt <custom-prompt>
   ```

### Advanced Usage

- Customize prompts to guide the response generation for extracted images.
- Choose different LLaVA models for varied response styles and complexities.

## Features

- **Image Extraction**: Extracts images from PDFs, with a focus on academic papers.
- **Figure Number Assignment**: Assigns unique figure numbers to each extracted image for reference.
- **Response Generation**: Uses the LLaVA model to generate descriptive responses for images.
- **Custom Prompts**: Supports customized prompts for specific image descriptions.

## Example

```bash
python pdf_image_analysis.py --paper_path "Genomics Papers/Rare and de novo variants.pdf" --prompt "Please describe the image"
```

## Dependencies

In addition to the core dependencies, the tool also requires:

- **PyMuPDF**: For PDF processing and image extraction.
- **pdfplumber**: To extract text surrounding images in PDFs for context.

These dependencies are included in the `pyproject.toml` file and will be installed during the project setup.

## License

Specify the licensing details here.

## Contributing

Provide guidelines for those who wish to contribute to the project.

---
