"""
Prompt generator ablation - basic prompt
"""
import sys
import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

BASIC_PROMPT = """
You are given two images: the first is a face image, the second is a pose image.
Describe the person's appearance combining facial features from the first image and body pose from the second image.
Write a single, cohesive sentence describing the person without referencing which image the details came from.
"""


class BasicPromptGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype="auto", 
            device_map="cuda" if torch.cuda.is_available() else None,
        )
        print("Model loaded successfully!")

    def analyze_image(self, face_image_path, pose_image_path, prompt_text):
        """Analyze two images (face + pose) with given prompt"""
        face_image = Image.open(face_image_path).convert("RGB")
        pose_image = Image.open(pose_image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=[formatted_prompt],
            images=[face_image, pose_image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        input_text = self.processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
        response = text[len(input_text):].strip()
        
        return response

    def extract_content(self, text):
        import re
        # Remove leading dashes and spaces
        text = re.sub(r'^-\s*', '', text.strip())
        # Extract content inside brackets
        matches = re.findall(r'\[(.*?)\]', text)
        if matches:
            return matches[0].strip()
        return text.strip()

    def generate_combined_prompt(self, face_img_path, pose_img_path):

        result_raw = self.analyze_image(face_img_path, pose_img_path, BASIC_PROMPT)
        result = self.extract_content(result_raw)
        print(f"✅ Basic prompt: {result}")

        return result

def main():
    from config import FACE_IMG, POSE_IMG
    generator = BasicPromptGenerator()
    combined_prompt = generator.generate_combined_prompt(FACE_IMG, POSE_IMG)
    print("🎉 Basic prompt generation completed!")

if __name__ == "__main__":
    main()