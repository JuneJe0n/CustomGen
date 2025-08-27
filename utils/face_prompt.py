"""
Category : gender, accesories
Model : llava-hf/llava-1.5-7b-hf
"""
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import re
import os
import csv
from typing import List

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


PROMPT_JUNE = """
You are a strict classifier. Classify the person in the picture on two properties :
1. Gender/Age (required): Classify if the person is [woman/girl/man/boy/baby]
2. Attribute : Classify if the person has [glasses/sunglasses/beard]. If none, output only the Gender/Age property.

Output Rules:
- Format your respose stricly as a single list 
- Do not add extra words, explanations, or categories.

Examples:
- [man]
- [woman]
- [boy, glasses]
- [man, beard]
"""


PROMPT_JEESOO = """
You are a strict classifier. For the given face image, output EXACTLY one lowercase line with 1 or 2 items, separated by a comma and a single space (", ").

- First item (required): one of {baby,boy,girl,man,woman}.
- Second item (optional): one of {sunglasses,glasses,beard}. If none, output only the first item.
- If multiple accessories are present, pick ONE using this priority: sunglasses > glasses > beard.

Hard rules:
- Output ONLY that line; no explanations, no extra words, no quotes, no trailing period.
- Do NOT output more than 2 items.
- Your entire output MUST match this regex:
^(baby|boy|girl|man|woman)(, (sunglasses|glasses|beard))?$

Examples:
baby
man, beard
woman, sunglasses
"""


CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_JUNE},
        ],
    },
]

class PersonAnalysisPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        print("Model loaded successfully!")

    @staticmethod
    def preprocess_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def build_inputs(self, image: Image.Image):
        prompt_text = self.processor.apply_chat_template(
            CONVERSATION,
            add_generation_prompt=True,
            tokenize=False,     
        )

        num_image_tokens = prompt_text.count(self.processor.tokenizer.special_tokens_map.get("image_token", "<image>"))
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True
        )
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def generate_analysis(self, image: Image.Image) -> str:
        inputs = self.build_inputs(image)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1].strip()
        return text

def main():
    print("Initializing Person Analysis Pipeline...")
    pipeline = PersonAnalysisPipeline()
    folder_path = "/data2/jeesoo/FFHQ/00000"
    csv_output_path = "/data2/jiyoon/custom/results/prompt/jiyoon_v1/face.csv"
    
    # Get all image files in the folder
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in supported_extensions]
    
    if not image_files:
        print("No image files found in the folder.")
        return
    
    print(f"Found {len(image_files)} image files. Processing...")
    
    # Prepare CSV output
    results = []
    
    for image_file in sorted(image_files):
        image_path = os.path.join(folder_path, image_file)
        try:
            image = pipeline.preprocess_image(image_path)
            raw_response = pipeline.generate_analysis(image)
            results.append({"image_file": image_file, "generated_prompt": raw_response})
            print(f"{image_file}: {raw_response}")
        except Exception as e:
            print(f"🚨 Error analyzing {image_file}: {e}")
            results.append({"image_file": image_file, "generated_prompt": "ERROR"})
    
    # Save results to CSV
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_file', 'generated_prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_output_path}")

if __name__ == "__main__":
    main()
