Here's an updated version of your `readme.md` file that includes the new `pdf_ext.py` program:

```markdown
# README for LLaVA PDF Image Analysis and Response Generation Tool

## Overview

LLaVA (Language Llama Variational Autoencoder) is a powerful tool designed to extract images and textual data such as tables and figures from PDF files, particularly from academic papers. It generates descriptive responses using advanced language and visual processing models. The tool can analyze images and text in scientific papers, identify and extract them along with their captions or nearby text, and use a pre-trained LLaVA model to generate responses based on custom prompts.

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
2. To extract images, run the script with the necessary arguments:
   ```bash
   python pdf_image_analysis.py --paper_path <path-to-pdf> --prompt <custom-prompt>
   ```
3. To extract textual data and tables, run the `pdf_ext.py` script:
   ```bash
   python pdf_ext.py --input_file <path-to-pdf> --output_file <path-to-output> --prompt <vision-language-prompt> --keyword <keyword> --convert_table <True/False> --convert_figure <True/False>
   ```

### Advanced Usage

- Customize prompts to guide the response generation for extracted images and textual content.
- Choose different LLaVA models for varied response styles and complexities.

## Features

- **Image and Text Extraction**: Extracts images, tables, and figures from PDFs, with a focus on academic papers.
- **Figure and Table Number Assignment**: Assigns unique numbers to each extracted image, table, and figure for reference.
- **Response Generation**: Uses the LLaVA model to generate descriptive responses for images and textual content.
- **Custom Prompts**: Supports customized prompts for specific image and text descriptions.

## Example

For image analysis:
```bash
python pdf_image_analysis.py --input_path "pdf/starGAN.pdf" --prompt "Please analyze the image in detail"
```

For text and table extraction:
```bash
python pdf_ext.py --input_path "pdf/starGAN.pdf" --output_path "output.txt" --prompt "Please analyze the image in detail" --keyword "GAN" --convert_table --convert_figure
```

## Dependencies

In addition to the core dependencies, the tool also requires:

- **PyMuPDF**: For PDF processing and image extraction.
- **pdfplumber**: To extract text and tables surrounding images in PDFs for context.

These dependencies are included in the `pyproject.toml` file and will be installed during the project setup.

## License

Specify the licensing details here.

## Contributing

Provide guidelines for those who wish to contribute to the project.

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main): the codebase we built upon.
- [pdf_ext.py](https://github.com/mingking2048/pdf-content-extractor): Our new tool for text and table extraction from PDFs.

---