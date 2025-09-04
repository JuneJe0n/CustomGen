#!/bin/bash

# Selective batch processing script for m5_ablation_4.py
# Usage: ./run_m5_ablation_4_selective.sh <face_folder> <pose_folder> <style_folder> <output_dir> [gpu_id]

set -e

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <face_folder> <pose_folder> <style_folder> <output_dir> [gpu_id]"
    echo "Example: $0 ./faces ./poses ./styles ./results 0"
    exit 1
fi

FACE_FOLDER="$1"
POSE_FOLDER="$2"
STYLE_FOLDER="$3"
OUTPUT_BASE_DIR="$4"
GPU_ID="${5:-0}"

# Validate folders exist
for folder in "$FACE_FOLDER" "$POSE_FOLDER" "$STYLE_FOLDER"; do
    if [ ! -d "$folder" ]; then
        echo "Error: Folder $folder does not exist"
        exit 1
    fi
done

# Get absolute paths
FACE_FOLDER=$(realpath "$FACE_FOLDER")
POSE_FOLDER=$(realpath "$POSE_FOLDER")
STYLE_FOLDER=$(realpath "$STYLE_FOLDER")
OUTPUT_BASE_DIR=$(realpath "$OUTPUT_BASE_DIR")

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Create temporary config file
TEMP_CONFIG=$(mktemp)
cp config.py "$TEMP_CONFIG"

# Define target folders
declare -a target_folders=(
    "00000_b_0_wikiart_021"
    "00000_b_0_wikiart_032"
    "00000_b_2_wikiart_015"
    "00003_b_4_wikiart_006"
    "00003_b_4_wikiart_008"
    "00003_b_6_wikiart_021"
    "00003_b_6_wikiart_032"
    "00003_b_6_wikiart_035"
    "00010_b_0_wikiart_032"
    "00010_b_0_wikiart_035"
    "00013_b_0_s12"
    "00013_b_0_wikiart_003"
    "00013_b_0_wikiart_006"
    "00013_b_0_wikiart_008"
    "00013_b_0_wikiart_011"
    "00013_b_0_wikiart_015"
    "00013_b_0_wikiart_016"
    "00013_b_0_wikiart_021"
    "00013_b_0_wikiart_032"
    "00013_b_0_wikiart_035"
    "00013_b_4_wikiart_035"
    "00013_b_6_s12"
    "00013_b_6_wikiart_003"
    "00013_b_6_wikiart_006"
    "00013_b_6_wikiart_008"
    "00013_b_6_wikiart_011"
    "00013_b_6_wikiart_015"
    "00013_b_6_wikiart_016"
    "00013_b_6_wikiart_021"
    "00013_b_6_wikiart_032"
    "00013_b_6_wikiart_035"
    "00026_b_0_s12"
    "00026_b_0_wikiart_003"
    "00026_b_0_wikiart_006"
    "00026_b_0_wikiart_008"
    "00026_b_0_wikiart_011"
    "00026_b_0_wikiart_015"
    "00026_b_0_wikiart_016"
    "00026_b_0_wikiart_021"
    "00026_b_0_wikiart_032"
    "00026_b_0_wikiart_035"
    "00026_b_2_s12"
    "00026_b_2_wikiart_003"
    "00026_b_2_wikiart_006"
    "00026_b_2_wikiart_008"
    "00026_b_2_wikiart_011"
    "00026_b_2_wikiart_015"
    "00026_b_2_wikiart_016"
    "00026_b_2_wikiart_021"
    "00026_b_2_wikiart_032"
    "00026_b_2_wikiart_035"
    "00026_b_3_s12"
    "00026_b_3_wikiart_003"
    "00026_b_3_wikiart_006"
    "00026_b_3_wikiart_008"
    "00026_b_3_wikiart_011"
    "00026_b_3_wikiart_015"
    "00026_b_3_wikiart_016"
    "00026_b_3_wikiart_021"
    "00026_b_3_wikiart_032"
    "00026_b_3_wikiart_035"
    "00026_b_4_s12"
    "00026_b_4_wikiart_003"
    "00026_b_4_wikiart_006"
    "00026_b_4_wikiart_008"
    "00026_b_4_wikiart_011"
    "00026_b_4_wikiart_015"
    "00026_b_4_wikiart_016"
    "00026_b_4_wikiart_021"
    "00026_b_4_wikiart_032"
    "00026_b_4_wikiart_035"
    "00026_b_6_s12"
    "00026_b_6_wikiart_003"
    "00026_b_6_wikiart_006"
    "00026_b_6_wikiart_008"
    "00026_b_6_wikiart_011"
    "00026_b_6_wikiart_015"
    "00026_b_6_wikiart_016"
    "00026_b_6_wikiart_021"
    "00026_b_6_wikiart_032"
    "00026_b_6_wikiart_035"
)

echo "Starting selective batch processing..."
echo "Face folder: $FACE_FOLDER"
echo "Pose folder: $POSE_FOLDER" 
echo "Style folder: $STYLE_FOLDER"
echo "Output base dir: $OUTPUT_BASE_DIR"
echo "GPU: $GPU_ID"
echo "Target folders: ${#target_folders[@]}"
echo ""

processed_count=0
skipped_count=0

# Process all combinations but only run for target folders
for face_file in "$FACE_FOLDER"/*; do
    if [[ -f "$face_file" && "$face_file" =~ \.(jpg|jpeg|png|bmp|tiff)$ ]]; then
        face_name=$(basename "$face_file")
        face_name_no_ext="${face_name%.*}"
        
        for pose_file in "$POSE_FOLDER"/*; do
            if [[ -f "$pose_file" && "$pose_file" =~ \.(jpg|jpeg|png|bmp|tiff)$ ]]; then
                pose_name=$(basename "$pose_file")
                pose_name_no_ext="${pose_name%.*}"
                
                for style_file in "$STYLE_FOLDER"/*; do
                    if [[ -f "$style_file" && "$style_file" =~ \.(jpg|jpeg|png|bmp|tiff)$ ]]; then
                        style_name=$(basename "$style_file")
                        style_name_no_ext="${style_name%.*}"
                        
                        # Create output directory name
                        outdir_name="${face_name_no_ext}_${pose_name_no_ext}_${style_name_no_ext}"
                        
                        # Check if this combination is in our target list
                        if [[ " ${target_folders[@]} " =~ " ${outdir_name} " ]]; then
                            processed_count=$((processed_count + 1))
                            output_dir="$OUTPUT_BASE_DIR/$outdir_name"
                            
                            echo "[$processed_count/${#target_folders[@]}] Processing: $outdir_name"
                            
                            # Create temporary config with current paths
                            cat > config.py << EOF
from pathlib import Path
from utils import PromptGenerator

# img paths
FACE_IMG  = Path("$face_file")
POSE_IMG  = Path("$pose_file")
STYLE_IMG = Path("$style_file")
OUTDIR    = Path("$output_dir")
OUTDIR.mkdir(parents=True, exist_ok=True)

# prompts
generator = PromptGenerator()
PROMPT = generator.generate_combined_prompt(FACE_IMG, POSE_IMG)
NEG = "(lowres, bad quality, watermark,strange limbs)"

# model paths
CN_HED     = "/data2/jiyoon/custom/ckpts/controlnet-union-sdxl-1.0"
CN_POSE    = "/data2/jiyoon/custom/ckpts/controlnet-openpose-sdxl-1.0"
BASE_SDXL  = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE_ENC  = "/data2/jiyoon/IP-Adapter/sdxl_models/image_encoder"
STYLE_IP   = "/data2/jiyoon/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

# params
COND_HED     = 0.8
COND_POSE    = 0.85
STYLE_SCALE  = 0.8
CFG, STEPS   = 7.0, 50
SEED         = 42
EOF
                            
                            # Run the script
                            if python m5_ablation_4.py --gpu "$GPU_ID"; then
                                echo "✅ Completed: $outdir_name"
                            else
                                echo "❌ Failed: $outdir_name"
                            fi
                            
                            echo ""
                        else
                            skipped_count=$((skipped_count + 1))
                        fi
                    fi
                done
            fi
        done
    fi
done

# Restore original config
mv "$TEMP_CONFIG" config.py

echo "Selective batch processing completed!"
echo "Processed: $processed_count target combinations"
echo "Skipped: $skipped_count non-target combinations"
echo "Results saved in: $OUTPUT_BASE_DIR"