#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def find_failed_folders(results_dir):
    """Find folders that are empty or missing 8_final_result.png"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return []
    
    failed_folders = []
    
    # Find all folders starting with "00000"
    for folder in results_path.iterdir():
        if folder.is_dir() and folder.name.startswith("00000"):
            # Check if folder is empty
            folder_contents = list(folder.iterdir())
            
            if not folder_contents:
                failed_folders.append(folder.name + " (empty)")
            else:
                # Check if 8_final_result.png exists
                final_result_file = folder / "8_final_result.png"
                if not final_result_file.exists():
                    failed_folders.append(folder.name + " (missing 8_final_result.png)")
    
    return sorted(failed_folders)

if __name__ == "__main__":
    results_dir = "/data2/jiyoon/custom/results/method5_ablation/ablation_4"
    
    failed = find_failed_folders(results_dir)
    
    if failed:
        print(f"Found {len(failed)} failed folders:")
        for folder in failed:
            print(folder)
    else:
        print("All folders are complete with 8_final_result.png")