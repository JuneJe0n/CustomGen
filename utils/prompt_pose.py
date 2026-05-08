"""
Prompt generator ablation - pose only prompt
"""
import sys
import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

POSE_PROMPT =  """
Please analyze the person in the picture. Provide a brief description of the pose of the person. Take carefull consider of the pose of the arms, legs and the overall body.

Format your response strictly as a single list.
Examples: 
- [Sitting]
- [Standing, arms crossed]
"""

# POSE_PROMPT =  """
# Please analyze the person in the picture. Provide a brief description of the pose of the person. Take carefull consider of the pose of the arms, legs and the overall body.

# Format your response strictly as a single list.
# Examples: 
# - [Sitting]
# - [Standing, arms crossed]
# """


class PoseOnlyPromptGenerator:
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

        # face_result_raw = self.analyze_image(face_img_path, FACE_PROMPT)
        # print(f"👶 Face prompt: {face_result_raw}")
        
        pose_result_raw = self.analyze_image(pose_img_path, POSE_PROMPT)
        # print(f"🕺 Pose prompt: {pose_result_raw}")
        
        # Extract clean content
        # face_result = self.extract_content(face_result_raw)
        pose_result = self.extract_content(pose_result_raw)
        
        # Combine results with comma
        prompt_pose = f"{pose_result}"
        print(f"✅ Pose only prompt: {pose_result}")
        
        return prompt_pose

def main():
    from config import POSE_IMG
    generator = PoseOnlyPromptGenerator()
    prompt = generator.generate_prompt(POSE_IMG)
    print("🎉 Pose only prompt generation completed!")

if __name__ == "__main__":
    main()