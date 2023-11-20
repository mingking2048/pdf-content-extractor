from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# liuhaotian/llava-v1.5-7b-lora
def run_llava_with_huggingface(prompt, image_file, model_path="liuhaotian/llava-v1.5-7b"):
    """tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )"""

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 1, 
        "top_p":1, 
        "max_new_tokens": 1024,  
        "num_beams": 1, 
    })()

    return eval_model(args)

# Example usage
prompt = "Please describe the image"
image_path = "test.png"
response = run_llava_with_huggingface(prompt, image_path)
