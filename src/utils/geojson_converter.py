import json
import argparse
import os

# --- Constants ---
MICRONS_PER_PIXEL = 0.242
MM_TO_MICRONS = 1000
PIXELS_PER_MM = MM_TO_MICRONS / MICRONS_PER_PIXEL

# Default classification properties (matching your example)
DEFAULT_CLASSIFICATION_NAME = "lymphocytes"
# Using the color from the second example object's properties
DEFAULT_CLASSIFICATION_COLOR = [244, 250, 88]
DEFAULT_OBJECT_TYPE = "annotation"

def mm_to_pixels(coord_mm):
    """Converts a coordinate from millimeters to pixels."""
    return coord_mm * PIXELS_PER_MM

def create_geojson_feature(point_data, box_size_px):
    """Creates a GeoJSON Feature dictionary for a single detection point."""

    point_name = point_data.get("name", "Unknown Point")
    point_coords_mm = point_data.get("point", [0, 0, 0])
    probability = point_data.get("probability", 0.0)

    if len(point_coords_mm) < 2:
        print(f"Warning: Skipping point '{point_name}' due to insufficient coordinates.")
        return None

    # Extract mm coordinates (ignore Z if present)
    x_mm, y_mm = point_coords_mm[0], point_coords_mm[1]

    # Convert center coordinates to pixels
    center_x_px = mm_to_pixels(x_mm)
    center_y_px = mm_to_pixels(y_mm)

    # Calculate bounding box pixel coordinates
    half_box = box_size_px / 2.0
    x_min = center_x_px - half_box
    y_min = center_y_px - half_box
    x_max = center_x_px + half_box
    y_max = center_y_px + half_box

    # Define polygon coordinates in GeoJSON format
    # [[[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]]
    coordinates = [
        [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
            [x_min, y_min] # Close the polygon
        ]
    ]

    # Create the feature dictionary
    feature = {
        "type": "Feature",
        "id": point_name, # Using point name as ID, adjust if needed
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        },
        "properties": {
            "objectType": DEFAULT_OBJECT_TYPE,
            "classification": {
                "name": DEFAULT_CLASSIFICATION_NAME,
                "color": DEFAULT_CLASSIFICATION_COLOR
            },
            "color": DEFAULT_CLASSIFICATION_COLOR, # Matching classification color
            "probability": probability,
            "center_mm": [x_mm, y_mm], # Optionally keep original coords
            "center_pixels": [center_x_px, center_y_px] # Optionally keep center px
        }
    }
    return feature

def convert_pipeline_output_to_geojson(input_json_path, output_geojson_path, box_size_px, prob_threshold=0.0):
    """Loads pipeline JSON, converts points to GeoJSON features, and saves."""

    print(f"Loading input JSON from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            pipeline_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the input file: {e}")
        return

    if "points" not in pipeline_data or not isinstance(pipeline_data["points"], list):
        print("Error: Input JSON does not contain a valid 'points' list.")
        return

    print(f"Processing {len(pipeline_data['points'])} points...")
    print(f"Using bounding box size: {box_size_px}x{box_size_px} pixels")
    print(f"Conversion rate: {PIXELS_PER_MM:.4f} pixels per mm ({MICRONS_PER_PIXEL} microns/pixel)")
    if prob_threshold > 0.0:
        print(f"Applying probability threshold: {prob_threshold}")

    geojson_features = []
    filtered_count = 0
    
    for point_data in pipeline_data["points"]:
        # Skip points below the probability threshold
        probability = point_data.get("probability", 0.0)
        if probability < prob_threshold:
            filtered_count += 1
            continue
            
        feature = create_geojson_feature(point_data, box_size_px)
        if feature:
            geojson_features.append(feature)

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} points below probability threshold {prob_threshold}")
    print(f"Including {len(geojson_features)} points in output")

    # Create the final GeoJSON FeatureCollection
    geojson_output = {
        "type": "FeatureCollection",
        "features": geojson_features
    }

    print(f"Saving GeoJSON output to: {output_geojson_path}")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_geojson_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_geojson_path, 'w') as f:
            json.dump(geojson_output, f, indent=4) # Use indent for readability
        print("Conversion successful!")
    except Exception as e:
        print(f"Error writing GeoJSON file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert detection pipeline JSON (mm coordinates) to GeoJSON (pixel coordinates)."
    )
    parser.add_argument(
        "input_json",
        help="Path to the input JSON file from the detection pipeline."
    )
    parser.add_argument(
        "output_geojson",
        help="Path to save the output GeoJSON file."
    )
    parser.add_argument(
        "--box_size",
        type=int,
        default=10, # Default box size of 10x10 pixels
        help="Size of the bounding box (width and height) in pixels. Default: 10"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.0, 
        help="Probability threshold (0.0-1.0). Only include detections with probability > threshold. Default: 0.0"
    )

    args = parser.parse_args()

    convert_pipeline_output_to_geojson(
        args.input_json,
        args.output_geojson,
        args.box_size,
        args.prob_threshold
    )