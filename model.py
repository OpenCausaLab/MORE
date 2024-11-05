from PIL import Image
import requests
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor,\
    AutoProcessor,\
     AutoTokenizer, AutoModelForCausalLM,\
    LlavaForConditionalGeneration
import torch
from more.utils.mplug_owl import MplugOwlForConditionalGeneration
from more.utils.mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
class Huggingface_Models:
    def __init__(self, model_name, path, device):
        self.device = torch.device(device)
        self.model_name = model_name
        self.torch_dtype = torch.float16
        if model_name == 'instructblip':
            self.processor = InstructBlipProcessor.from_pretrained(path)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(path, orch_dtype=torch.float16).to(self.device)
        elif model_name == 'mplug':
            self.torch_dtype = torch.bfloat16
            self.model = MplugOwlForConditionalGeneration.from_pretrained(path, torch_dtype=self.torch_dtype).to(self.device)
            image_processor = MplugOwlImageProcessor.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
        elif model_name == 'llava_vicuna':
            self.model = LlavaForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to(self.device)
            self.processor = AutoProcessor.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        elif model_name == 'qwen':
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", trust_remote_code=True).eval()


    def vqa(self, img_id, prompt, max_new_tokens=20):
        if isinstance(img_id, str):
            if img_id.startswith('http'):
                image = Image.open(requests.get(img_id, stream=True).raw)
            else:
                image = Image.open(img_id)
        else:
            image = img_id

        if self.model_name == 'mplug':
            image, prompt = [image], [prompt]
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            res = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=5)
            generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        elif self.model_name == 'qwen':
            query = self.tokenizer.from_list_format([
                {'image': img_id},
                {'text': prompt},
            ])
            inputs = self.tokenizer(query, return_tensors='pt')
            inputs = inputs.to(self.device)
            pred = self.model.generate(**inputs)
            res = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
            generated_text = res.split("The best answer is: (")[-1]
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if self.model_name == 'llava_vicuna':
                generated_text = generated_text.split("The best answer is: (")[-1]

        return generated_text.strip()
