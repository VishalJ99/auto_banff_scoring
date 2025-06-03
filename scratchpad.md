# Project Scratchpad

## Project Overview
Automatic Banff scoring system using InstanSeg for cell detection on kidney transplant biopsy WSIs.

## Current Tasks
- [ ] Handle resolution for in-house data (read actual resolution from SVS metadata)
- [ ] Fix coordinate scaling to use actual resolution: `coords * (0.5/actual_mpp)`
- [ ] Fix mm conversion to use actual resolution: `pixel * actual_mpp / 1000`
- [ ] Test inference on challenge data to verify coordinate scaling works correctly
- [ ] Investigate impact of resolution mismatch on detection quality

## Decisions Log

### Resolution Issue Discovery (03/06/2025)
- **Problem**: instanseg_inference.py hardcodes pixel size to 0.242 μm/pixel but actual SVS files have 0.263 μm/pixel
- **Impact**: InstanSeg processes images thinking cells are ~8.5% smaller than reality
- **Current workaround**: GeoJSON converter uses same hardcoded value, making overlays appear correct spatially
- **Root cause**: Script adapted from challenge dataset where all images had 0.242 mpp

### InstanSeg Model Native Resolution (03/06/2025)
- **Key Finding**: InstanSeg model was trained at 0.5 μm/pixel resolution
- **Confirmed by**: Loading model directly and checking `instanseg_model.instanseg.pixel_size` = 0.5
- **Impact**: All InstanSeg processing happens at 0.5 μm/pixel, NOT at input image resolution

### Coordinate Transformation NOT A BUG (03/06/2025)
- Line 154 in instanseg_inference.py scales coordinates by 2.066× (0.5/0.242)
- **This is CORRECT**: Converts from 0.5 μm/pixel (model space) to original image space
- **rescale_output=False**: Means labels stay at 0.5 μm/pixel, not original resolution
- The scaling is necessary and intentional for proper coordinate transformation

## Notes & Context

### How InstanSeg Resolution Works
1. Model trained at 0.5 μm/pixel resolution (native resolution)
2. When pixel_size=0.242 is passed:
   - InstanSeg calculates scale_factor = 0.242/0.5 = 0.484
   - Input image downscaled by 0.484× to match model's resolution
3. Processing happens at 0.5 μm/pixel
4. With rescale_output=False:
   - Labels remain at 0.5 μm/pixel (model's native resolution)
   - Coordinates from centroids_from_lab are in 0.5 μm/pixel space
5. Coordinate scaling (0.5/0.242) converts back to original image space

### Key Files
- `src/instanseg_inference.py`: Main inference script with hardcoded resolution
- `geojson_converter.py`: Converts detection JSON to GeoJSON format
- Detection JSONs: Store coordinates in mm with z=0.242 (hardcoded pixel size)

## Quick References
- InstanSeg hardcoded pixel size: 0.24199951445730394 μm/pixel
- Typical SVS resolution: 0.262719 μm/pixel (example from validation set)
- Scaling factor in bug: 0.5 / 0.242 ≈ 2.066

## Why Perfect Overlays Despite Wrong Resolution

### Key Insight
InstanSeg doesn't rescale images based on pixel_size parameter - it processes at native resolution. The pixel_size only affects internal model decisions (expected cell sizes), NOT coordinate transformations.

### Why Overlays Work
1. InstanSeg rescales input image from 0.242 to 0.5 μm/pixel (downscale by 0.484×)
2. Processing happens at 0.5 μm/pixel (model's native resolution)
3. Coordinates correctly scaled by 2.066× to convert from 0.5 back to 0.242 μm/pixel space
4. GeoJSON converter reverses this scaling to get pixel coordinates for plotting
5. Both use same hardcoded pixel size (0.242), creating self-consistent system

### What's Actually Wrong
- Detection QUALITY compromised: InstanSeg thinks cells are ~8.5% smaller than reality
- Affects segmentation boundaries, cell separation, size filtering
- Cell POSITIONS still correct in pixel space

### Challenge vs In-house Data
- Challenge: 0.242 mpp images → told 0.242 mpp ✓ (optimal detection)
- In-house: 0.263 mpp images → told 0.242 mpp ✗ (degraded detection)

The coordinate bug was likely introduced for the challenge but is masked by the conversion pipeline.

## Key Confusion Points for Testing

### Resolution Understanding CORRECTED
- When `rescale_output=False`, InstanSeg outputs labels at 0.5 μm/pixel (NOT original resolution)
- `labels`: At 0.5 μm/pixel resolution (e.g., 484×484 for a 1000×1000 @ 0.242 μm/pixel image)
- `tensor`: Also rescaled to 0.5 μm/pixel (e.g., 484×484)
- `get_masked_patches(labels, tensor)` receives same-sized inputs ✓
- The 2.066× scaling (0.5/0.242) converts coordinates from 0.5 μm/pixel space back to original image space

### Coordinate Flow CORRECTED
1. `centroids_from_lab(labels)` → returns coords in 0.5 μm/pixel space (484×484)
2. Line 143: `coords * (0.5/0.242)` → scales from 0.5 μm/pixel to original resolution space
3. Line 143: `+ bbox_native[0][::-1]` → adds bbox origin offset in original resolution
4. Lines 181-182: Convert to mm using 0.242 μm/pixel (should use actual resolution)

### What is coords_scaled?
- After line 143, coords_scaled represents cell positions in the full WSI coordinate system
- bbox_native[0] is the top-left corner of the extracted region in WSI pixels
- Adding bbox_native shifts from "coords within the bbox" to "coords in full WSI"

### Updated Understanding
The 2.066× scaling is NOT a bug - it's necessary to convert coordinates from 0.5 μm/pixel space (where InstanSeg operates) back to the original image resolution. 

How it works:
1. InstanSeg model was trained at 0.5 μm/pixel resolution
2. When we pass pixel_size=0.242, InstanSeg calculates scale_factor = 0.242/0.5 = 0.484
3. Input image is downscaled by 0.484× before processing
4. With rescale_output=False, labels stay at 0.5 μm/pixel (model's native resolution)
5. Coordinates from centroids_from_lab are in 0.5 μm/pixel space
6. The 2.066× scaling (0.5/0.242) converts these back to original image space

The real issues are:
1. Hardcoded 0.242 μm/pixel doesn't match actual SVS files (0.263 μm/pixel)
2. Conversion to mm uses wrong resolution (0.242 instead of actual)

### Fix Needed
1. Read actual resolution from SVS metadata
2. Use actual resolution for coordinate scaling: `coords * (0.5/actual_mpp)`
3. Use actual resolution for mm conversion: `pixel * actual_mpp / 1000`

### Critical Question
If the model outputs coords at wrong scale, the JSON would have cell positions that don't match reality. Need to verify if the mm coordinates in the output JSON match expected cell locations.