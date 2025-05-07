import json
import argparse
import os

MICRONS_PER_PIXEL = 0.242  # Adjust if needed
PIXELS_PER_MM = 1000 / MICRONS_PER_PIXEL
DEFAULT_CLASSIFICATION_NAME = "inflammatory"
DEFAULT_CLASSIFICATION_COLOR = [255, 0, 0]
DEFAULT_OBJECT_TYPE = "annotation"

def mm_to_pixels(coord_mm):
    return coord_mm * PIXELS_PER_MM

def create_geojson_feature(point_data, box_size_px):
    name = point_data.get("name", "unknown")
    coords_mm = point_data["point"]
    prob = point_data.get("probability", 0.0)

    if len(coords_mm) < 2:
        return None

    x_px = mm_to_pixels(coords_mm[0])
    y_px = mm_to_pixels(coords_mm[1])
    half = box_size_px / 2

    coords = [
        [
            [x_px - half, y_px - half],
            [x_px - half, y_px + half],
            [x_px + half, y_px + half],
            [x_px + half, y_px - half],
            [x_px - half, y_px - half],
        ]
    ]

    return {
        "type": "Feature",
        "id": name,
        "geometry": {
            "type": "Polygon",
            "coordinates": coords,
        },
        "properties": {
            "objectType": DEFAULT_OBJECT_TYPE,
            "classification": {
                "name": DEFAULT_CLASSIFICATION_NAME,
                "color": DEFAULT_CLASSIFICATION_COLOR,
            },
            "color": DEFAULT_CLASSIFICATION_COLOR,
            "probability": prob,
            "center_mm": coords_mm[:2],
            "center_pixels": [x_px, y_px],
        },
    }

def convert_inflamm_json_to_geojson(input_path, output_path, box_size_px=10, prob_thresh=0.0):
    with open(input_path) as f:
        data = json.load(f)

    if "points" not in data:
        raise ValueError("Input JSON must contain a 'points' list.")

    features = []
    for pt in data["points"]:
        if pt.get("probability", 0.0) >= prob_thresh:
            feat = create_geojson_feature(pt, box_size_px)
            if feat:
                features.append(feat)

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=4)
    print(f"✅ Converted {len(features)} points → saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Input detection JSON")
    parser.add_argument("output_geojson", help="Output GeoJSON for QuPath")
    parser.add_argument("--box_size", type=int, default=10)
    parser.add_argument("--prob_threshold", type=float, default=0.0)
    args = parser.parse_args()

    convert_inflamm_json_to_geojson(args.input_json, args.output_geojson, args.box_size, args.prob_threshold)
