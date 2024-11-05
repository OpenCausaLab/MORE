import json
import os
import pathlib
import time
import argparse
import os.path
from model import Huggingface_Models
from tqdm import tqdm
import datetime
from wandb.sdk.data_types.trace_tree import Trace
import google.generativeai as genai
import wandb
from openai import OpenAI
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from more.utils.utils import *
model_path = {
    "instructblip": "instructblip-vicuna-13b",
    "mplug": "mplug-owl-llama-7b",
    "llava_vicuna": "llava-1.5-13b-hf",
    "qwen": "Qwen-VL",
}


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
def run_model_vqa(model, model_name, img_path, prompt, ground_truth=None, max_new_token=20):
    token_usage = {}
    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    try:
        if "gpt" in model_name.lower():
            if model_name == 'gpt-4v':
                model_name = 'gpt-4-turbo'
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": "Output your choice (option name, e.g., A, B, etc.) first."
                        }]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_new_token,
                temperature=0.0,
            )
            response = res.choices[0].message.content

        elif 'claude' in model_name:
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            if model_name == 'claude_opus':
                model_name = 'claude-3-opus-20240229'
            elif model_name == 'claude_sonnet':
                model_name = 'claude-3-sonnet-20240229'
            message = client.messages.create(
                model=model_name,
                max_tokens=max_new_token,
                temperature=0.0,
                system="Output your choice (option name, e.g., A, B, etc.) first.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_image(img_path),
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            response = message.content[0].text

        elif 'gemini' in model_name:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-pro')
            cookie_picture = {
                'mime_type': 'image/jpeg',
                'data': pathlib.Path(img_path).read_bytes()
            }
            response = model.generate_content(["System Instruction: output your choice (option name, e.g., A, B, etc.) first.\n"+prompt, cookie_picture],
                                              generation_config=genai.types.GenerationConfig(
                                                      # Only one candidate for now.
                                                      candidate_count=1,
                                                      max_output_tokens=max_new_token,
                                                      temperature=0))
            time.sleep(5.0)
            response = response.text

        else:
            response = model.vqa(img_path, prompt, max_new_token)
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        status = "success"
        status_message = (None,)
        response_text = response
        if response_text[0] == '(':
            response_text = response_text[1:]
    except Exception as e:
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        status = "error"
        status_message = str(e)
        response_text = " "
    root_span = Trace(
        name="root_span",
        kind="llm",
        status_code=status,
        status_message=status_message,
        metadata={
            "token_usage": token_usage,
            "model_name": model_name,
        },
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        inputs={"query": prompt},
        outputs={"response": response_text,
                 "ground_truth": ground_truth},
    )
    root_span.log(name="llm_trace")
    return response_text


def evaluate(args, name):
    if args.model in ['instructblip', 'mplug', 'llava_vicuna', 'qwen']:
        model = Huggingface_Models(args.model, model_path[args.model], args.device)
    else:
        model = None
    eval_data = json.load(open(args.cache_dir))
    acc = 0
    two_hop_acc, three_hop_acc = 0, 0
    two_hop_count, three_hop_count = 0, 0
    vis_count, lan_count, mm_count = 0, 0, 0
    with open("result/" + name + ".jsonl", "w", encoding="utf-8") as f:
        for item_i, item in tqdm(enumerate(eval_data)):
            options = item['options']
            answer_index = item['correct_option_idx']
            option_text = convert_options(options)
            item['prompt'] = f"Question: {item['question']}\nChoose from the following options:\n{option_text}\n"
            if args.model == 'llava_vicuna':
                item['prompt'] = prompt_answer_with_input(item['prompt'], "mcq", args.model)
            else:
                item['prompt'] += "The best answer is: ("

            prompt = item['prompt']
            img_path = os.path.join(args.image_path, item["image_id"] + '.jpg')
            if img_path.endswith("JPEG"):
                img_path = img_path.replace("JPEG", "jpg")
            ground_truth = chr(ord('A') + answer_index)
            response_text = run_model_vqa(model, args.model, img_path, prompt, ground_truth, 200)
            item["response"] = response_text
            item["ground_truth"] = ground_truth
            f.write(json.dumps(item) + '\n')
            if response_text[0] == ground_truth:
                if item["hop"] == 2:
                    two_hop_acc += 1
                elif item["hop"] == 3:
                    three_hop_acc += 1
                acc += 1
            if item["hop"] == 2:
                two_hop_count += 1
            elif item["hop"] == 3:
                three_hop_count += 1
            choice = ord(item["response"][0]) - ord('A')
            if choice < 0 or choice > 3:
                continue
            elif item["options"][choice] in item["vision_option"]:
                vis_count += 1
            elif item["options"][choice] == item["language_option"]:
                lan_count += 1
            elif item["options"][choice] in item["semantic_misleading_option"]:
                mm_count += 1
    print("Overall Accuracy is: %.02f\n" % (acc / len(eval_data)))
    wandb.log({'Accuracy': acc / len(eval_data)})

    wandb.log({'Two Hop Accuracy': two_hop_acc / two_hop_count})
    wandb.log({'Three Hop Accuracy': three_hop_acc / three_hop_count})
    wandb.log({'Vis Num': vis_count, 'Lan Num': lan_count, 'MM Num': mm_count})

    print(two_hop_count, ",", three_hop_count, "\n")
    print(vis_count, " ", lan_count, " ", mm_count)
    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    json.dump(eval_data, open("result/" + name + ".json", "w"), indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default='MORE_val.json', type=str)
    parser.add_argument("--image_path", default='./InfoSeek', type=str)
    parser.add_argument("--output_dir", default='./output', type=str)
    parser.add_argument("--dataset", default='MORE', type=str)
    parser.add_argument("--model", default='gpt-4o', type=str)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    set_seed()
    name = args.dataset + '_' + args.model
    wandb.init(project="MORE", config=args, name=name)
    evaluate(args, name)


if __name__ == "__main__":
    main()