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


PROMPT_TEXT = (
    """
    Please analyze the person in the picture. Provide:
    1. Gender/Age category: Identify it the person is a woman, girl, man, boy, or baby. Return at least one.
    2. Attributes: Identify if they are wearing or have any of glasses, sunglasses, beard, or none. Return at least one (none counts as an option).
    Do not return an empty list. You must return at least one from each option.

    Format your response strictly as a single list.
    Examples: 
    - [man, sunglasses]
    - [woman]
    - [boy, glasses]
    - [man, beard]
    """
)

CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_TEXT},
        ],
    },
]

ALLOWED_GENDER_AGE = {"woman", "girl", "man", "boy", "baby"}
ALLOWED_ACCESSORIES = {"glasses", "sunglasses", "none"}

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

    def parse_response(self, response: str) -> List[str]:
        if response.strip() == "[]":
            return []
        matches = re.findall(r"\[([^\]]+)\]", response)
        if not matches:
            return []
        
        parts = [p.strip().lower() for p in matches[0].split(",") if p.strip()]
        gender_age = None
        accessory = None
        for p in parts:
            if p in ALLOWED_GENDER_AGE and gender_age is None:
                gender_age = p
            elif p in ALLOWED_ACCESSORIES and accessory is None:
                accessory = p
        
        if gender_age:
            if accessory and accessory != "none":
                return [gender_age, accessory]
            else:
                return [gender_age]
        return []


    def analyze_person(self, image_path: str) -> List[str]:
        image = self.preprocess_image(image_path)
        raw = self.generate_analysis(image)
        return self.parse_response(raw)

def main():
    print("Initializing Person Analysis Pipeline...")
    pipeline = PersonAnalysisPipeline()
    folder_path = "/data2/jeesoo/FFHQ/00000"
    csv_output_path = "/data2/jiyoon/custom/results/prompt/face.csv"
    
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
            result = pipeline.analyze_person(image_path)
            prompt = ", ".join(result) if result else ""
            results.append({"image_file": image_file, "generated_prompt": prompt})
            print(f"{image_file}: {result}")
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
