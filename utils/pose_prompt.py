"""
Category : pose
Model : llava-hf/llava-1.5-7b-hf
"""
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import re
import csv
import os
import glob
from typing import List

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


PROMPT_TEXT = (
    """
    Please analyze the person in the picture. Provide a brief description of the pose of the person. Take carefull consider of the pose of the arms, legs and the overall body.

    Format your response strictly as a single list.
    Examples: 1
    - [Sitting]
    - [Standing, arms crossed]
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
        
        # Take only the first match since we assume one person
        parts = [p.strip() for p in matches[0].split(",") if p.strip()]
        return parts


    def analyze_person(self, image_path: str) -> List[str]:
        image = self.preprocess_image(image_path)
        raw = self.generate_analysis(image)
        return self.parse_response(raw)
    
    def process_folder(self, folder_path: str, output_csv: str = "pose_results.csv"):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return
        
        print(f"Found {len(image_files)} images to process...")
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'response'])
            
            for i, image_path in enumerate(image_files, 1):
                filename = os.path.basename(image_path)
                print(f"Processing {i}/{len(image_files)}: {filename}")
                
                try:
                    result = self.analyze_person(image_path)
                    response = str(result) if result else "[]"
                    writer.writerow([filename, response])
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    writer.writerow([filename, f"Error: {str(e)}"])
        
        print(f"Results saved to {output_csv}")

def main():
    print("Initializing Person Analysis Pipeline...")
    pipeline = PersonAnalysisPipeline()
    folder_path = "/data2/jiyoon/custom/data/pose"
    output_csv = "/data2/jiyoon/custom/results/prompt/jiyoon/pose.csv"
    try:
        pipeline.process_folder(folder_path, output_csv)
    except Exception as e:
        print(f"🚨 Error during analysis: {e}")

if __name__ == "__main__":
    main()
