"""
Prompt generator ablation - only face prompt
"""
import sys
import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

FACE_PROMPT = """
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

# POSE_PROMPT =  """
# Please analyze the person in the picture. Provide a brief description of the pose of the person. Take carefull consider of the pose of the arms, legs and the overall body.

# Format your response strictly as a single list.
# Examples: 
# - [Sitting]
# - [Standing, arms crossed]
# """


class FaceOnlyPromptGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype="auto", 
            device_map="cuda" if torch.cuda.is_available() else None,
        )
        print("Model loaded successfully!")

    def analyze_image(self, image_path, prompt_text):
        """Analyze image with given prompt"""
        image = Image.open(image_path).convert("RGB")
        
        conversation = [
            {
                "role": "user",
                "content": [
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
            images=image,
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

    def generate_prompt(self, face_img_path):

        face_result_raw = self.analyze_image(face_img_path, FACE_PROMPT)
        # print(f"👶 Face prompt: {face_result_raw}")
        
        # pose_result_raw = self.analyze_image(pose_img_path, POSE_PROMPT)
        # print(f"🕺 Pose prompt: {pose_result_raw}")
        
        # Extract clean content
        face_result = self.extract_content(face_result_raw)
        # pose_result = self.extract_content(pose_result_raw)
        
        # Combine results with comma
        prompt_face = f"{face_result}"
        print(f"✅ Face only prompt: {face_result}")
        
        return prompt_face

def main():
    from config import FACE_IMG
    generator = PromptGenerator()
    combined_prompt = generator.generate_prompt(FACE_IMG)
    print("🎉 Face only prompt generation completed!")

if __name__ == "__main__":
    main()