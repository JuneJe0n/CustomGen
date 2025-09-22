#!/bin/bash

# Activate conda environment
source /home/jiyoon/.miniconda3/etc/profile.d/conda.sh
conda activate instantstyle

# Set GPU to use (GPU 2 is free according to gpustat)
export CUDA_VISIBLE_DEVICES=1

# Directories containing the images
FACE_DIR="/data2/jiyoon/custom/data/ablation/face/baby"
POSE_DIR="/data2/jiyoon/custom/data/ablation/pose/baby"
STYLE_DIR="/data2/jiyoon/custom/data/ablation/style"
OUTPUT_DIR="/data2/jiyoon/custom/results/final/infer_4"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter for progress tracking
counter=0
total_combinations=0

echo "Using GPU 5 (CUDA_VISIBLE_DEVICES=5)"
echo "Output directory: $OUTPUT_DIR"

# Count total combinations first
echo "Counting total combinations..."

# Check directories and count files
echo "Checking face directory: $FACE_DIR"
face_count=$(find "$FACE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
echo "  Found $face_count face files"

echo "Checking pose directory: $POSE_DIR"
pose_count=$(find "$POSE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
echo "  Found $pose_count pose files"

echo "Checking style directory: $STYLE_DIR"
style_count=$(find "$STYLE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
echo "  Found $style_count style files"

if [[ $face_count -eq 0 ]]; then
    echo "ERROR: No image files found in face directory"
    exit 1
fi
if [[ $pose_count -eq 0 ]]; then
    echo "ERROR: No image files found in pose directory"
    exit 1
fi
if [[ $style_count -eq 0 ]]; then
    echo "ERROR: No image files found in style directory"
    exit 1
fi

for face_img in "$FACE_DIR"/*; do
    if [[ -f "$face_img" ]]; then
        for pose_img in "$POSE_DIR"/*; do
            if [[ -f "$pose_img" ]]; then
                for style_img in "$STYLE_DIR"/*; do
                    if [[ -f "$style_img" ]]; then
                        ((total_combinations++))
                    fi
                done
            fi
        done
    fi
done

echo "Total combinations to generate: $total_combinations"
echo "Starting generation..."
echo ""

# Generate all combinations
for face_img in "$FACE_DIR"/*; do
    if [[ -f "$face_img" ]]; then
        # Extract face filename without extension
        face_basename=$(basename "$face_img")
        face_name="${face_basename%.*}"
        
        for pose_img in "$POSE_DIR"/*; do
            if [[ -f "$pose_img" ]]; then
                # Extract pose filename without extension
                pose_basename=$(basename "$pose_img")
                pose_name="${pose_basename%.*}"
                
                for style_img in "$STYLE_DIR"/*; do
                    if [[ -f "$style_img" ]]; then
                        # Extract style filename without extension
                        style_basename=$(basename "$style_img")
                        style_name="${style_basename%.*}"
                        
                        # Create output filename
                        output_filename="${face_name}_${pose_name}_${style_name}.png"
                        output_path="$OUTPUT_DIR/$output_filename"
                        
                        ((counter++))
                        echo "[$counter/$total_combinations] Generating: $output_filename"
                        echo "  Face: $face_basename"
                        echo "  Pose: $pose_basename" 
                        echo "  Style: $style_basename"
                        
                        # Run the inference script with current combination
                        echo "Running: python infer_5.py --face_img \"$face_img\" --pose_img \"$pose_img\" --style_img \"$style_img\" --output_path \"$output_path\""
                        
                        python infer_5.py \
                            --face_img "$face_img" \
                            --pose_img "$pose_img" \
                            --style_img "$style_img" \
                            --output_path "$output_path" 2>&1
                        
                        exit_code=$?
                        if [ $exit_code -eq 0 ]; then
                            echo "  ✅ Success: $output_filename"
                        else
                            echo "  ❌ Failed: $output_filename (exit code: $exit_code)"
                            # Continue with next combination even if one fails
                        fi
                        echo ""
                    fi
                done
            fi
        done
    fi
done

echo "================================"
echo "All combinations completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Total generated: $counter files"
echo "================================"