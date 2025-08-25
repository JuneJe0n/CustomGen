"""
Code that extracts gender, age, and accessories from face pic
Base model : LLaVA v1.5
"""
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import re
from typing import Dict, List, Optional

PROMPT = """
USER: <image>
Please analyze this image and identify any people present. For each person you see, provide:

1. Gender/Age category: Choose from: woman, girl, man, boy, baby
2. Accessories: Identify if they are wearing: glasses, sunglasses, or none

Format your response as a list.
Example output :
- [woman]
- [man, sunglasses]

If no people are visible, respond with: []

A:
"""

CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    },
]

class PersonAnalysisPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load LLaVA 1.5 processor and model
        self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",  # Fixed: was using undefined 'model_name'
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        print("Model loaded successfully!")
        
        self.prompt = PROMPT
        self.conversation = CONVERSATION

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess the input image."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            # If it's already a PIL Image
            image = image_path.convert("RGB")
        return image

    def generate_analysis(self, image: Image.Image) -> str:
        """Generate analysis using LLaVA 1.5 model."""
        # Update conversation with actual image
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.replace("USER: <image>", "").replace("A:", "").strip()},
                ],
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images=[image]  # Add the actual image
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode the response
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("ASSISTANT:")[-1].strip()
        
        return assistant_response

    def parse_response(self, response: str) -> Dict:
        """Parse the model response for list format like [woman], [man, sunglasses]."""
        try:
            response = response.strip()
            
            # Check if no people detected
            if response.lower() in ["[]", "none", "no people", "no one", ""]:
                return {"result": "[]", "people": []}
            
            # Find all list items like [woman] or [man, sunglasses]
            list_pattern = r'\[([^\]]+)\]'
            matches = re.findall(list_pattern, response)
            
            people = []
            result_lines = []
            
            for i, match in enumerate(matches, 1):
                parts = [part.strip() for part in match.split(',')]
                
                if len(parts) >= 1:
                    category = parts[0].strip()
                    accessories = [acc.strip() for acc in parts[1:] if acc.strip()]
                    
                    people.append({
                        "person_id": i,
                        "gender_age": category,
                        "accessories": accessories
                    })
                    
                    # Format for simple result
                    if accessories:
                        result_lines.append(f"{category}, {', '.join(accessories)}")
                    else:
                        result_lines.append(category)
            
            # If no matches found, try fallback parsing
            if not matches:
                return self.fallback_parse(response)
            
            result_str = "\n".join(result_lines) if result_lines else "none"
            return {"result": result_str, "people": people}
            
        except Exception as e:
            print(f"Parsing error: {e}")
            return self.fallback_parse(response)

    def fallback_parse(self, response: str) -> Dict:
        """Fallback parser for non-standard responses."""
        result = {"result": "none", "people": []}
        
        # Define valid categories
        gender_age_keywords = ["woman", "girl", "man", "boy", "baby"]
        accessories_keywords = ["glasses", "sunglasses"]
        
        response_lower = response.lower()
        
        # Check if any person is mentioned
        person_indicators = ["person", "individual", "woman", "man", "girl", "boy", "baby", "people"]
        has_person = any(indicator in response_lower for indicator in person_indicators)
        
        if has_person:
            found_gender = None
            found_accessories = []
            
            # Find gender/age category
            for keyword in gender_age_keywords:
                if keyword in response_lower:
                    found_gender = keyword
                    break
            
            # Find accessories
            for accessory in accessories_keywords:
                if accessory in response_lower:
                    found_accessories.append(accessory)
            
            # If we found a gender category, create a person entry
            if found_gender:
                result["people"].append({
                    "person_id": 1,
                    "gender_age": found_gender,
                    "accessories": found_accessories
                })
                
                # Format result string
                if found_accessories:
                    result["result"] = f"{found_gender}, {', '.join(found_accessories)}"
                else:
                    result["result"] = found_gender
        
        return result

    def get_simple_result(self, image_path: str) -> str:
        """
        Get simple comma-separated result format.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            String in format: "boy, glasses" or "girl" or "none"
        """
        result = self.analyze_person(image_path)
        
        if result["status"] != "success":
            return "error"
        
        return result["analysis"]["result"]

    def analyze_person(self, image_path: str) -> Dict:
        """
        Main method to analyze a person in an image.
        
        Args:
            image_path: Path to the input image or PIL Image object
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load and preprocess image
            image = self.preprocess_image(image_path)
            
            # Generate analysis
            raw_response = self.generate_analysis(image)
            
            # Parse response
            parsed_result = self.parse_response(raw_response)
            
            # Add metadata
            result = {
                "status": "success",
                "model": "LLaVA-1.5",
                "raw_response": raw_response,
                "analysis": parsed_result
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "model": "LLaVA-1.5",
                "error": str(e),
                "analysis": {"result": "error", "people": []}
            }

# Utility functions for the pipeline
def validate_result(result: Dict) -> bool:
    """Validate the analysis result structure."""
    if "analysis" not in result:
        return False
    
    if "people" not in result["analysis"]:
        return False
    
    for person in result["analysis"]["people"]:
        required_fields = ["person_id", "gender_age", "accessories"]
        if not all(field in person for field in required_fields):
            return False
        
        # Validate gender_age category
        valid_categories = ["woman", "girl", "man", "boy", "baby"]
        if person["gender_age"] not in valid_categories:
            return False
        
        # Validate accessories
        valid_accessories = ["glasses", "sunglasses"]
        if not isinstance(person["accessories"], list):
            return False
        
        for accessory in person["accessories"]:
            if accessory not in valid_accessories:
                return False
    
    return True

def format_results(result: Dict) -> str:
    """Format results for human-readable output."""
    if result["status"] != "success":
        return f"Error: {result.get('error', 'Unknown error')}"
    
    # Return the simple result format
    return result["analysis"]["result"]

# Example usage
def main():
    print("Initializing LLaVA 1.5 Person Analysis Pipeline...")
    pipeline = PersonAnalysisPipeline()
    
    # Example usage
    image_path = "/data2/jeesoo/FFHQ/00000/00000.png"
    
    try:
        print(f"Analyzing image: {image_path}")
        
        # Get simple result
        simple_result = pipeline.get_simple_result(image_path)
        print(f"\nSimple Result: {simple_result}")
        
        # Get full analysis
        result = pipeline.analyze_person(image_path)
        
        # Validate result
        if validate_result(result):
            print("✓ Valid result structure")
        else:
            print("⚠ Result structure validation failed")
        
        # Print formatted results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        print(format_results(result))
        
        # Print raw response for debugging
        print(f"\nRaw model response: {result.get('raw_response', 'N/A')}")
        
        # Show expected format examples
        print("\n" + "="*50)
        print("EXPECTED OUTPUT FORMATS")
        print("="*50)
        print("Model should output:")
        print("- [woman]")
        print("- [man, sunglasses]")
        print("- [boy, glasses]")
        print("- []  (if no people)")
        print(f"\nParsed to simple format: '{simple_result}'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()