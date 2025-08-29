"""
Category : gender, accessories
Model : 
- Salesforce/blip2-opt-2.7b 
- blip2-flan-t5-xl
"""
import torch
from PIL import Image
import os
import csv
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_ID = "Salesforce/blip2-opt-2.7b"  # Smaller, faster
MODEL_ID = "Salesforce/blip2-flan-t5-xl"  # Larger, potentially better performance

GENDER_PROMPT = "What is the gender and age category of the person? Answer with one word: baby, boy, girl, man, or woman."

ACCESSORY_PROMPTS = {
    "glasses": "Is the person wearing glasses? Answer yes or no.",
    "sunglasses": "Is the person wearing sunglasses? Answer yes or no.", 
    "beard": "Does the person have a beard? Answer yes or no."
}

class PersonAnalysisPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load BLIP-2 model and processor
        self.processor = Blip2Processor.from_pretrained(MODEL_ID)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        print("BLIP-2 model loaded successfully!")

    @staticmethod
    def preprocess_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def ask_question(self, image: Image.Image, question: str) -> str:
        """Ask a single question about the image"""
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=3,
                temperature=1.0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        response = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
        return response.lower()

    def classify_gender_age(self, image: Image.Image) -> str:
        """Classify gender and age category"""
        response = self.ask_question(image, GENDER_PROMPT)
        
        # Extract the classification from response
        gender_age_options = ['baby', 'boy', 'girl', 'man', 'woman']
        for option in gender_age_options:
            if option in response:
                return option
        
        # Fallback - try to parse common responses
        if 'male' in response or 'boy' in response:
            return 'man' if 'adult' in response or 'old' in response else 'boy'
        elif 'female' in response or 'girl' in response:
            return 'woman' if 'adult' in response or 'old' in response else 'girl'
        elif 'infant' in response or 'child' in response:
            return 'baby'
        
        return 'unknown'

    def check_accessories(self, image: Image.Image) -> list:
        """Check for accessories in order of priority"""
        accessories = []
        
        # Check in priority order: sunglasses > glasses > beard
        for accessory, prompt in ACCESSORY_PROMPTS.items():
            response = self.ask_question(image, prompt)
            
            if 'yes' in response or accessory in response:
                accessories.append(accessory)
                break  # Only return the highest priority accessory found
        
        return accessories

    def generate_analysis(self, image: Image.Image) -> str:
        """Generate complete analysis following the required format"""
        try:
            # Get gender/age classification
            gender_age = self.classify_gender_age(image)
            
            if gender_age == 'unknown':
                return 'unknown'
            
            # Get accessories
            accessories = self.check_accessories(image)
            
            # Format output according to the strict requirements
            if accessories:
                return f"{gender_age}, {accessories[0]}"
            else:
                return gender_age
                
        except Exception as e:
            print(f"Error in analysis: {e}")
            return 'error'

def main():
    print("Initializing BLIP-2 Person Analysis Pipeline...")
    pipeline = PersonAnalysisPipeline()
    
    folder_path = "/data2/jiyoon/custom/data/face"
    csv_output_path = "/data2/jiyoon/custom/results/prompt/jiyoon/BLIP2/face.csv"
    
    # Get all image files in the folder
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in supported_extensions]
        
    print(f"Found {len(image_files)} image files. Processing...")

    results = []
    
    for i, image_file in enumerate(sorted(image_files)):
        image_path = os.path.join(folder_path, image_file)
        try:
            image = pipeline.preprocess_image(image_path)
            raw_response = pipeline.generate_analysis(image)
            results.append({"image_file": image_file, "generated_prompt": raw_response})
            print(f"[{i+1}/{len(image_files)}] {image_file}: {raw_response}")
        except Exception as e:
            print(f"🚨 Error analyzing {image_file}: {e}")
            results.append({"image_file": image_file, "generated_prompt": "ERROR"})
    
    # Save results to CSV
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_file', 'generated_prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_output_path}")
    print(f"Processed {len(results)} images total")

if __name__ == "__main__":
    main()