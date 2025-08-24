"""
Code that extracts gender, age, and accesories from face pic

Base model : LLaVA v1.5
"""
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import re
from typing import Dict, List, Optional

class PersonAnalysisPipeline:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize the LLaVA 1.5-based person analysis pipeline.
        
        Args:
            model_name: HuggingFace model identifier for LLaVA 1.5
                       Options: "llava-hf/llava-1.5-7b-hf" or "llava-hf/llava-1.5-13b-hf"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading LLaVA 1.5 model: {model_name}")
        
        # Load LLaVA 1.5 processor and model
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        
        print("Model loaded successfully!")
        
        # Define the structured prompt for person analysis
        self.analysis_prompt = """USER: <image>
Please analyze this image and identify any people present. For each person you see, provide:

1. Gender/Age category: Choose from: woman, girl, man, boy, baby
2. Accessories: Identify if they are wearing: glasses, sunglasses, or none

Format your response as JSON:
{
  "people": [
    {
      "person_id": 1,
      "gender_age": "woman/girl/man/boy/baby",
      "accessories": ["glasses", "sunglasses"] or []
    }
  ]
}

If no people are visible, respond with: {"people": []}

ASSISTANT:"""

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
        # Prepare inputs for LLaVA 1.5
        inputs = self.processor(
            text=self.analysis_prompt,
            images=image,
            return_tensors="pt"
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
        """Parse the model response and return simple comma-separated format."""
        try:
            # Clean the response
            response = response.strip()
            
            # Check if no people detected
            if response.lower() in ["none", "no people", "no one", ""]:
                return {"result": "none", "people": []}
            
            # Parse comma-separated format
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            people = []
            
            for i, line in enumerate(lines, 1):
                if ',' in line:
                    # Has accessories: "boy, glasses"
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 2:
                        category = parts[0]
                        accessories = [acc.strip() for acc in parts[1:]]
                        people.append({
                            "person_id": i,
                            "gender_age": category,
                            "accessories": accessories
                        })
                else:
                    # No accessories: "girl"
                    category = line.strip()
                    people.append({
                        "person_id": i,
                        "gender_age": category,
                        "accessories": []
                    })
            
            return {"result": response, "people": people}
            
        except Exception as e:
            print(f"Parsing error: {e}")
            return {"result": response, "people": []}

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

    def fallback_parse(self, response: str) -> Dict:
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
        """Fallback parser for non-JSON responses."""
        result = {"people": []}
        
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
        
        return result

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

# Batch processing version
class BatchPersonAnalysisPipeline(PersonAnalysisPipeline):
    def analyze_batch(self, image_paths: List[str], max_batch_size: int = 4) -> List[Dict]:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths: List of image paths
            max_batch_size: Maximum number of images to process at once
        """
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(image_paths), max_batch_size):
            batch_paths = image_paths[i:i + max_batch_size]
            
            for image_path in batch_paths:
                try:
                    result = self.analyze_person(image_path)
                    result["image_path"] = image_path
                    results.append(result)
                except Exception as e:
                    results.append({
                        "image_path": image_path,
                        "status": "error",
                        "model": "LLaVA-1.5",
                        "error": str(e),
                        "analysis": {"result": "error", "people": []}
                    })
        
        return results

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
    
    people = result["analysis"]["people"]
    
    if not people:
        return "No people detected in the image."
    
    output = []
    for person in people:
        accessories_str = ", ".join(person["accessories"]) if person["accessories"] else "None"
        output.append(f"Person {person['person_id']}: {person['gender_age']}, Accessories: {accessories_str}")
    
    return "\n".join(output)

# Example usage
def main():
    """Example usage of the LLaVA 1.5 person analysis pipeline."""
    # Initialize the pipeline
    print("Initializing LLaVA 1.5 Person Analysis Pipeline...")
    pipeline = PersonAnalysisPipeline()
    
    # Example usage
    image_path = "/data2/jeesoo/FFHQ/00000/00000.png"
    
    try:
        print(f"Analyzing image: {image_path}")
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
        
        # Print raw JSON for debugging
        print("\n" + "="*50)
        print("RAW JSON OUTPUT")
        print("="*50)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error during analysis: {e}")

# Testing function
def test_pipeline():
    """Test the pipeline with various scenarios."""
    pipeline = PersonAnalysisPipeline()
    
    test_cases = [
        "test_woman_with_glasses.jpg",
        "test_man_with_sunglasses.jpg",
        "test_child.jpg",
        "test_baby.jpg",
        "test_no_person.jpg"
    ]
    
    for test_image in test_cases:
        print(f"\nTesting: {test_image}")
        try:
            result = pipeline.analyze_person(test_image)
            print(format_results(result))
        except FileNotFoundError:
            print(f"Test image {test_image} not found - skipping")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()