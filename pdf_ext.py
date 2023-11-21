import pdfplumber
import re
from pdf_image_analysis import image_analysis
import pdf_image_analysis
import argparse

def find_table_titles(page):
    titles = []
    text = page.extract_text()
    if text:
        for line in text.split('\n'):
            # Regular expression match for titles starting with "Table", followed by numbers and a period
            if re.match(r'Table \d+\.', line):
                titles.append(line)
    return titles


def match_figure_title(line, n):
    match = re.match(r'(?:Figure|Fig\.)\s?(\d+)', line, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        if 1 <= number <= n:
            return number
    return None

def find_figure(page, n):
    titles = []
    text = page.extract_text()
    if text:
        for line in text.split('\n'):
            matched_number = match_figure_title(line, n)
            if matched_number is not None:
                titles.append(int(matched_number))
    return titles


def extract_text_and_tables(args):
    text_content = []
    tables_content = []
    current_title = None

    if args.convert_figure:
        # image analysis
        fig_dic = image_analysis(args)
        # print(fig_dic)

    with pdfplumber.open(args.input_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)

            if args.convert_table:
                # Search for possible table titles
                titles = find_table_titles(page)

                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if titles:
                        current_title = titles.pop(0)  
                    text_content.append((current_title, table))

            if args.convert_figure:
                fig_list = find_figure(page, len(fig_dic)+1)
                for i in fig_list:
                    if str(i) in fig_dic:
                        text_content.append([str(i), fig_dic[str(i)]])

    

    return text_content

def parse_args():
    parser = argparse.ArgumentParser(description='Extract text and tables from a PDF file.')
    parser.add_argument('--input_path', type=str, help='Path to the input PDF file')
    parser.add_argument('--output_path', type=str, help='Path to the output text file')
    parser.add_argument('--convert_table', action='store_true', help='convert the image into a formatted document?')
    parser.add_argument('--convert_figure', action='store_true', help='convert an image into text?')
    return pdf_image_analysis.parse_args(parser)


def main():
    
    args = parse_args()
    text = extract_text_and_tables(args)
    
    with open(args.output_path, 'w') as f:
        for item in text:
            print(item, file=f)



if __name__ == "__main__":
    main()