import json
import argparse
import os
from typing import List, Dict, Any

# --- Constants ---
MICRONS_PER_PIXEL = 0.262719  # Based on your existing script
MM_TO_MICRONS = 1000
PIXELS_PER_MM = MM_TO_MICRONS / MICRONS_PER_PIXEL

# Default classification properties for inflammatory cells
DEFAULT_CLASSIFICATION_NAME = "inflammatory"
DEFAULT_CLASSIFICATION_COLOR = [255, 0, 0]  # Red color for inflammatory cells
DEFAULT_OBJECT_TYPE = "annotation"

def create_geojson_feature(coord_data: Dict[str, Any], box_size_px: int = 10) -> Dict[str, Any]:
    """Creates a GeoJSON Feature dictionary for a single inflammatory cell detection."""
    
    # Extract coordinates and metadata
    x_px = coord_data.get("x", 0.0)
    y_px = coord_data.get("y", 0.0)
    cell_type = coord_data.get("type", "inflammatory")
    probability = coord_data.get("prob", 0.0)
    
    # Calculate bounding box pixel coordinates
    half_box = box_size_px / 2.0
    x_min = x_px - half_box
    y_min = y_px - half_box
    x_max = x_px + half_box
    y_max = y_px + half_box
    
    # Define polygon coordinates in GeoJSON format
    # QuPath expects [[[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]]
    coordinates = [
        [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
            [x_min, y_min]  # Close the polygon
        ]
    ]
    
    # Create the feature dictionary
    feature = {
        "type": "Feature",
        "id": f"{cell_type}_{x_px}_{y_px}",  # Unique ID based on position
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        },
        "properties": {
            "objectType": DEFAULT_OBJECT_TYPE,
            "classification": {
                "name": cell_type,
                "color": DEFAULT_CLASSIFICATION_COLOR
            },
            "color": DEFAULT_CLASSIFICATION_COLOR,
            "probability": probability,
            "center_pixels": [x_px, y_px]
        }
    }
    return feature

def extract_inflammatory_coordinates(input_json_path: str, 
                                   output_geojson_path: str, 
                                   box_size_px: int = 10, 
                                   prob_threshold: float = 0.0,
                                   cell_types: List[str] = None) -> None:
    """
    Extracts inflammatory cell coordinates from pipeline output JSON and converts to GeoJSON.
    
    Args:
        input_json_path: Path to the input JSON file from the detection pipeline
        output_geojson_path: Path to save the output GeoJSON file
        box_size_px: Size of the bounding box (width and height) in pixels
        prob_threshold: Probability threshold (0.0-1.0). Only include detections with probability > threshold
        cell_types: List of cell types to include. If None, includes all inflammatory types
    """
    
    if cell_types is None:
        cell_types = ["inflammatory", "lymphocyte", "monocyte"]
    
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
    
    # Check if coordinates section exists
    if "coordinates" not in pipeline_data:
        print("Error: Input JSON does not contain a 'coordinates' section.")
        return
    
    coordinates_data = pipeline_data["coordinates"]
    all_coords = []
    
    # Extract coordinates for each cell type
    for cell_type in cell_types:
        if cell_type in coordinates_data:
            cell_coords = coordinates_data[cell_type]
            print(f"Found {len(cell_coords)} {cell_type} coordinates")
            all_coords.extend(cell_coords)
        else:
            print(f"Warning: No coordinates found for cell type '{cell_type}'")
    
    if not all_coords:
        print("No coordinates found for any of the specified cell types.")
        return
    
    print(f"Processing {len(all_coords)} total coordinates...")
    print(f"Using bounding box size: {box_size_px}x{box_size_px} pixels")
    
    if prob_threshold > 0.0:
        print(f"Applying probability threshold: {prob_threshold}")
    
    geojson_features = []
    filtered_count = 0
    
    for coord_data in all_coords:
        # Skip coordinates below the probability threshold
        probability = coord_data.get("prob", 0.0)
        if probability < prob_threshold:
            filtered_count += 1
            continue
        
        feature = create_geojson_feature(coord_data, box_size_px)
        if feature:
            geojson_features.append(feature)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} coordinates below probability threshold {prob_threshold}")
    print(f"Including {len(geojson_features)} coordinates in output")
    
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
            json.dump(geojson_output, f, indent=2)  # Pretty print for readability
        print("Conversion successful!")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total coordinates processed: {len(all_coords)}")
        print(f"- Coordinates included: {len(geojson_features)}")
        print(f"- Coordinates filtered: {filtered_count}")
        
    except Exception as e:
        print(f"Error writing GeoJSON file: {e}")

def print_data_summary(input_json_path: str) -> None:
    """Print a summary of the data structure and available cell types."""
    
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Data summary for: {input_json_path}")
        print("="*50)
        
        # Print slide information
        if "slide" in data:
            print(f"Slide ID: {data['slide']}")
        
        # Print cell counts
        cell_types = ["inflammatory", "lymphocyte", "monocyte"]
        for cell_type in cell_types:
            if cell_type in data:
                print(f"{cell_type.capitalize()} count: {data[cell_type]}")
        
        # Print normalized metrics if available
        if "normalised" in data:
            print("\nNormalized metrics:")
            for key, value in data["normalised"].items():
                print(f"  {key}: {value:.3f}")
        
        # Print coordinate availability
        if "coordinates" in data:
            print(f"\nCoordinate data available:")
            for cell_type, coords in data["coordinates"].items():
                print(f"  {cell_type}: {len(coords)} coordinates")
                if coords:
                    print(f"    Sample coordinate: x={coords[0]['x']}, y={coords[0]['y']}, prob={coords[0]['prob']:.3f}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract inflammatory cell coordinates from pipeline JSON and convert to GeoJSON for QuPath."
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
        default=10,
        help="Size of the bounding box (width and height) in pixels. Default: 10"
    )
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold (0.0-1.0). Only include detections with probability > threshold. Default: 0.0"
    )
    parser.add_argument(
        "--cell_types",
        nargs="+",
        default="inflammatory",
        help="Cell types to include in the output. Default: inflammatory lymphocyte monocyte"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of the input data without conversion."
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print_data_summary(args.input_json)
    else:
        extract_inflammatory_coordinates(
            args.input_json,
            args.output_geojson,
            args.box_size,
            args.prob_threshold,
            args.cell_types
        )