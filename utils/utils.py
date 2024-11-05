import base64
import torch
import random
import numpy as np
import io

def encode_image(image_path):
    if isinstance(image_path, str):
        if image_path.startswith('http'):
            return image_path
        else:
            with open(image_path, "rb") as image_file:
                b64 = base64.b64encode(image_file.read()).decode('utf-8')
            return f"{b64}"
    else:
        buffered = io.BytesIO()
        image_path.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"{b64}"

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def convert_options(options):
    formatted_options = []

    for i, option in enumerate(options):
        formatted_option = f"({chr(65 + i)}). {option}"
        formatted_options.append(formatted_option)

    result = '\n'.join(formatted_options)
    return result

phrase_answer_multiple_choice = "The best answer is: ("
phrase_answer_open_ended = "The best short answer is:"

is_chat_model = True  # TODO: so far for all models used here

def prompt_answer(c_task, E_INST):
    if c_task == 'open_ended':
        return f"""{E_INST if is_chat_model else ''}{phrase_answer_open_ended}\n"""
    else:
        return f"""{E_INST if is_chat_model else ''}{phrase_answer_multiple_choice}"""


def prompt_answer_with_input(inputt, c_task, model_name="llava_vicuna"):
    if "llava_vicuna" == model_name:
        system_prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
        B_INST_IMG, B_INST, E_INST = f"{system_prompt} USER: <image>\n", "USER:\n", "\nASSISTANT:\n"
    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet.")

    return f"""{B_INST_IMG if is_chat_model else ''}{inputt}{prompt_answer(c_task, E_INST)}"""
