"""
Code that extracts gender, age, and accesories from face pic

Base model : LLaVA v1.5
"""

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import json
import re
from typing import Dict, List, Optional

class PersonAnalysisPipeline:
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        Initialize the LLaVA-based person analysis pipeline.
        
        Args:
            model_name: HuggingFace model identifier for LLaVA
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        # Define the structured prompt for person analysis
        self.analysis_prompt = """
Please analyze this image and provide information about any people you see. For each person, identify:
1. Gender category: woman, girl, man, boy, or baby
2. Accessories: glasses, sunglasses, or none

Respond in the following JSON format:
{
  "people": [
    {
      "person_id": 1,
      "gender_age": "woman/girl/man/boy/baby",
      "accessories": ["glasses", "sunglasses"] or []
    }
  ]
}

If no people are detected, respond with: {"people": []}
"""

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess the input image."""
        image = Image.open(image_path).convert("RGB")
        return image

    def generate_analysis(self, image: Image.Image) -> str:
        """Generate analysis using LLaVA model."""
        # Prepare inputs
        inputs = self.processor(
            text=self.analysis_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode the response
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the model's response (after the prompt)
        response_start = generated_text.find(self.analysis_prompt) + len(self.analysis_prompt)
        model_response = generated_text[response_start:].strip()
        
        return model_response

    def parse_response(self, response: str) -> Dict:
        """Parse the model response and extract structured information."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback parsing if JSON format is not perfect
                return self.fallback_parse(response)
        except json.JSONDecodeError:
            return self.fallback_parse(response)

    def fallback_parse(self, response: str) -> Dict:
        """Fallback parser for non-JSON responses."""
        result = {"people": []}
        
        # Look for gender/age keywords
        gender_age_keywords = ["woman", "girl", "man", "boy", "baby"]
        accessories_keywords = ["glasses", "sunglasses"]
        
        found_gender = None
        found_accessories = []
        
        response_lower = response.lower()
        
        for keyword in gender_age_keywords:
            if keyword in response_lower:
                found_gender = keyword
                break
        
        for accessory in accessories_keywords:
            if accessory in response_lower:
                found_accessories.append(accessory)
        
        if found_gender:
            result["people"].append({
                "person_id": 1,
                "gender_age": found_gender,
                "accessories": found_accessories
            })
        
        return result

    def analyze_person(self, image_path: str) -> Dict:
        """
        Main method to analyze a person in an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        
        # Generate analysis
        raw_response = self.generate_analysis(image)
        
        # Parse response
        parsed_result = self.parse_response(raw_response)
        
        # Add metadata
        result = {
            "status": "success",
            "raw_response": raw_response,
            "analysis": parsed_result
        }
        
        return result

# Example usage and testing
def main():
    # Initialize the pipeline
    pipeline = PersonAnalysisPipeline()
    
    # Example usage
    image_path = "path/to/your/image.jpg"
    
    try:
        result = pipeline.analyze_person(image_path)
        
        print("Analysis Results:")
        print(json.dumps(result, indent=2))
        
        # Extract key information
        people = result["analysis"]["people"]
        if people:
            for person in people:
                print(f"\nPerson {person['person_id']}:")
                print(f"  Category: {person['gender_age']}")
                print(f"  Accessories: {', '.join(person['accessories']) if person['accessories'] else 'None'}")
        else:
            print("No people detected in the image.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

# Advanced pipeline with batch processing
class BatchPersonAnalysisPipeline(PersonAnalysisPipeline):
    def analyze_batch(self, image_paths: List[str]) -> List[Dict]:
        """Analyze multiple images in batch."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.analyze_person(image_path)
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "status": "error",
                    "error": str(e)
                })
        
        return results

# Fine-tuning data preparation (if you want to fine-tune the model)
def prepare_training_data():
    """
    Example function to prepare training data for fine-tuning.
    You would need a dataset of images with corresponding labels.
    """
    training_examples = []
    
    # Example format for training data
    example = {
        "image": "path/to/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "Please analyze this image and identify the people, their gender/age category, and accessories."
            },
            {
                "from": "gpt",
                "value": '{"people": [{"person_id": 1, "gender_age": "woman", "accessories": ["glasses"]}]}'
            }
        ]
    }
    
    training_examples.append(example)
    return training_examples

if __name__ == "__main__":
    main()