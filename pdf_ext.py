import pdfplumber
import re
from pdf_image_analysis import image_analysis
import argparse

def find_table_titles(page):
    titles = []
    text = page.extract_text()
    if text:
        for line in text.split('\n'):
            # 正则表达式匹配 "Table" 开头，后跟数字和点的标题
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


def extract_text_and_tables(pdf_path, convert_table=True, convert_fig=True, prompt=None, variant=None):
    text_content = []
    tables_content = []
    current_title = None

    if convert_fig:
        # image
        fig_dic = image_analysis(paper_path=pdf_path, prompt=prompt, variant=variant)
        print(fig_dic)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 提取文本
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)

            if convert_table:
                # 寻找可能的表格标题
                titles = find_table_titles(page)

                # 提取表格
                page_tables = page.extract_tables()
                for table in page_tables:
                    if titles:
                        current_title = titles.pop(0)  
                    text_content.append((current_title, table))

            if convert_fig:
                fig_list = find_figure(page, len(fig_dic)+1)
                for i in fig_list:
                    if str(i) in fig_dic:
                        text_content.append([str(i), fig_dic[str(i)]])

    

    return text_content



def main():
    parser = argparse.ArgumentParser(description='Extract text and tables from a PDF file.')
    parser.add_argument('input_file', type=str, help='Path to the input PDF file')
    parser.add_argument('output_file', type=str, help='Path to the output text file')
    parser.add_argument('prompt', default=None, type=str, help='vision language prompt')
    parser.add_argument('variant', default=None, type=str, help='variant')
    parser.add_argument('convert_table', type=bool, help='convert the image into a formatted document?')
    parser.add_argument('convert_figure', type=bool, help='convert an image into text?')
    args = parser.parse_args()

    text = extract_text_and_tables(args.input_file, convert_table=args.convert_table, convert_fig=args.convert_figure, prompt=args.prompt, variant=args.variant)
    
    with open(args.output_file, 'w') as f:
        for item in text:
            print(item, file=f)



if __name__ == "__main__":
    main()
