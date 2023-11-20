import argparse
import torch
from genericpath import exists
import fitz  # PyMuPDF
import pdfplumber
import os
import re
from pathlib import Path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
import copy
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image


class PDFImageExtractor:
    def __init__(self, paper_path):
        self.paper_path = Path(paper_path)

    def find_figure_number(self, image_rect, text, assigned_figures):
        nearest_figure_number = None
        min_distance = float('inf')
        figure_number_pattern = re.compile(r'(?:Figure|Fig\.)\s?(\d+)', re.IGNORECASE)
    
        for match in figure_number_pattern.finditer(text):
            fig_num = match.group(1)
            if fig_num in assigned_figures:
                continue
    
            text_position = (image_rect.x0, image_rect.y0)  # Placeholder for text position
            distance = min(abs(image_rect.y0 - text_position[1]), abs(text_position[1] - image_rect.y1))
    
            if distance < min_distance:
                nearest_figure_number = fig_num
                min_distance = distance
    
        if nearest_figure_number:
            assigned_figures.add(nearest_figure_number)
        return nearest_figure_number

    def extract_images_with_captions(self):
        output_folder = Path("extracted_images") / (self.paper_path.stem + "_images")
        output_folder.mkdir(parents=True, exist_ok=True)
    
        images_captions_dict = {}
        assigned_figure_numbers = set()
    
        with fitz.open(self.paper_path) as doc, pdfplumber.open(self.paper_path) as pdf:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pdf_page = pdf.pages[page_num]
                entire_page_text = pdf_page.extract_text()
    
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    image_info = doc.extract_image(xref)
                    img_bytes = image_info["image"]
                    img_rect = fitz.Rect(img[1:5])
    
                    figure_number = self.find_figure_number(img_rect, entire_page_text, assigned_figure_numbers)
    
                    if figure_number:
                        image_filename = f"{figure_number}.png"
                        image_path = output_folder / image_filename
                        with image_path.open('wb') as img_file:
                            img_file.write(img_bytes)
    
                        images_captions_dict[figure_number] = str(image_path)

        return images_captions_dict

class LLaVAResponseGenerator:
    def __init__(self, temperature, top_p, max_new_tokens, num_beams, model_path="liuhaotian/llava-v1.5-7b"):
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        self.args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": self.model_name,
            "conv_mode": None,
            "sep": ",",
            "temperature": temperature, 
            "top_p": top_p, 
            "max_new_tokens": max_new_tokens,  
            "num_beams": num_beams,
        })()    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.args.model_path, self.args.model_base, self.model_name
        )
        
    def image_parser(self, args):
        out = args.image_file.split(args.sep)
        return out


    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    
    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out    
        
        
    def generate_response(self, prompt, image_file):  
        args = copy.deepcopy(self.args)
        args.image_file = image_file
        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
    
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode
    
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        image_files = self.image_parser(args)
        images = self.load_images(image_files)
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
    
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
    
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
    
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
    
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

   
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA PDF Image Analysis and Response Generation")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b", choices=["liuhaotian/llava-v1.5-7b", "liuhaotian/llava-v1.5-13b"], help="Model path for LLaVA")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for response generation")
    parser.add_argument("--top_p", type=float, default=1, help="Top P for response generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--paper_path", type=str, help="Path to the PDF paper")
    parser.add_argument("--prompt", type=str, default="Please describe the image", help="Prompt for image description")

    return parser.parse_args()

def image_analysis(paper_path, prompt=None, variant=None):
    args = parse_args()
    if prompt is None:
        if variant is None:
            prompt = "Please describe the image"
        else:
            prompt = f"Is there any evidence related to {variant} on this chart?"
    extractor = PDFImageExtractor(paper_path)
    images_captions = extractor.extract_images_with_captions()

    generator = LLaVAResponseGenerator(args.temperature, args.top_p, args.max_new_tokens, args.num_beams, args.model_path)
    figure_descriptions = {}
    for fig_num, img_path in images_captions.items():
        response = generator.generate_response(prompt, img_path)
        figure_descriptions[fig_num] = response
    return figure_descriptions

def main():
    args = parse_args()
    
    paper_dir = Path('pdf')
    paper_path = paper_dir / 'starGAN.pdf'
    args.paper_path = paper_path

    extractor = PDFImageExtractor(args.paper_path)
    images_captions = extractor.extract_images_with_captions()

    generator = LLaVAResponseGenerator(args.temperature, args.top_p, args.max_new_tokens, args.num_beams, args.model_path)
    for fig_num, img_path in images_captions.items():
        response = generator.generate_response(args.prompt, img_path)
        print(f"Figure {fig_num}: {response}")

if __name__ == "__main__":
    main()
    
# environment:
# conda create -n llava python=3.10 -y
# conda activate llava
# pip install --upgrade pip  # enable PEP 660 support
# pip install -e . 
# pip install -r requirements.txt    
# script:
# python pdf_image_analysis.py --paper_path ${paper_path} --prompt ${prompt}
