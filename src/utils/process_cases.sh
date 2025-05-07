#!/bin/bash

# Process SVS files for automatic Banff scores
# This script runs patch extraction and InstanSeg inference on a list of SVS files

# Create output directory

# Process TI=3 cases
echo "Processing TI=3 cases..."

# List of TI=3 SVS files
ti3_files=(
  "/vol/biomedic3/histopatho/win_share/2025-01-02/anon_2bb2f1fb-2a23-49cf-a632-79c704ebc454.svs"
  "/vol/biomedic3/histopatho/win_share/2025-01-02/anon_6200f5cd-e8a8-489e-b7ee-34deb71a02e3.svs"
  "/vol/biomedic3/histopatho/win_share/2025-01-02/anon_0c454ff5-93eb-4c7a-98c2-274b86fae957.svs"
  "/vol/biomedic3/histopatho/win_share/2025-01-02/anon_d02d7e14-23ac-4f9e-bab7-99d8a8aeabf0.svs"
  "/vol/biomedic3/histopatho/win_share/2025-01-02/anon_3a4bff04-357b-490f-b8f6-dffb9f371829.svs"
)

# Process TI=0 cases
echo "Processing TI=0 cases..."

# List of TI=0 SVS files
ti0_files=(
  "/vol/biomedic3/histopatho/win_share/2024-07-15/anon_acb24ec7-4d6f-4004-9ecc-0cfb647eb962.svs"
  "/vol/biomedic3/histopatho/win_share/2024-07-15/anon_1c31838a-d6d8-4a0b-960e-138c73bc3dc5.svs"
  "/vol/biomedic3/histopatho/win_share/2024-07-15/anon_6bd013fb-ffd6-44bf-ba8a-e50055cc928c.svs"
  "/vol/biomedic3/histopatho/win_share/2024-07-15/anon_6c68a4f7-1bd0-470a-a408-d476c4117698.svs"
  "/vol/biomedic3/histopatho/win_share/2024-07-15/anon_47ad2dbb-296d-4295-b98d-d6b993a2f5aa.svs"
)

# Combine all files for processing
all_files=("${ti3_files[@]}" "${ti0_files[@]}")

# Process each file
for svs_path in "${all_files[@]}"; do
  # Extract filename without path and extension
  filename=$(basename "$svs_path" .svs)
  
  echo "Processing $filename..."
  
  # Create output directory for this file
  output_dir="data/TI_cases_output/$filename"
  mkdir -p "$output_dir"
  
  # Step 1: Run patch extraction
  echo "Running patch extraction for $filename..."
  python src/utils/patch_extractor.py "$svs_path"
  
  # Step 2: Run inference
  echo "Running inference for $filename..."
  python src/instanseg_inference.py \
    --wsi_path "$svs_path" \
    --output_dir "$output_dir" \
    --bbox_file bbox_coordinates.txt \
    --model_dir /data2/vj724/automatic_banff_scores/models
  
  echo "Completed processing for $filename"
  echo "----------------------------------------"
done

echo "All files processed successfully!"