# dynamic_cells_route.py - Enhanced cell detection with dynamic parameter calculation

from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
import cv2
import logging
import io
import boto3
from utilities.auth_utility import protected
from utilities.logging_utility import ActivityLogger
from db.models import Sample
from PIL import Image
import os
import tempfile
import psutil
logger = logging.getLogger(__name__)

dynamic_cells_route_bp = Blueprint("dynamic_cells", __name__, url_prefix="dynamic-cells")

# PARAMETERS you can tweak
MAX_RESULTS = 500
LARGE_IMAGE_PIXEL_THRESHOLD = 7_500_000 
# ----------------------
# Helpers
# ----------------------

from scipy import ndimage
def _calculate_properties_iterative(polygons, original_bgr_image):
    """
    Calculates properties by processing each polygon individually on small cropped patches.
    This is the most memory-efficient method for sparse polygons on large images.
    """
    all_properties = []
    
    for polygon in polygons:
        if not polygon or len(polygon) < 3:
            continue

        try:
            poly_np = np.array(polygon, dtype=np.int32)
            
            # 1. Get the bounding box of the single polygon
            x, y, w, h = cv2.boundingRect(poly_np)
            if w == 0 or h == 0:
                continue

            # 2. Extract a tiny BGR patch from the original image
            bgr_patch = original_bgr_image[y:y+h, x:x+w]

            # 3. Create a tiny mask ONLY for the patch
            shifted_poly = poly_np - [x, y]
            patch_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(patch_mask, [shifted_poly], 255)

            # 4. Convert ONLY the tiny patch to HSV
            # --- FIX IS HERE ---
            hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)

            # 5. Calculate mean HSV on the patch. cv2.mean is highly optimized.
            mean_hsv = cv2.mean(hsv_patch, mask=patch_mask)

            # 6. Calculate geometric properties
            area = cv2.contourArea(poly_np)
            roundness = calculate_roundness(poly_np)
            M = cv2.moments(poly_np)
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (x + w//2, y + h//2)
            
            all_properties.append({
                'area': area,
                'roundness': roundness,
                'bbox': [x, y, x+w, y+h],
                'centroid': (cx, cy),
                'h_mean': mean_hsv[0],
                's_mean': mean_hsv[1],
                'v_mean': mean_hsv[2]
            })
        except Exception as e:
            logger.error(f"Could not process a polygon iteratively: {e}")
            continue
            
    return all_properties

    
def _calculate_polygon_properties_vectorized(polygons, full_hsv):
    """
    Calculates area, roundness, and mean HSV for a list of polygons using vectorized operations.
    This is highly memory-efficient as it avoids creating large intermediate arrays.
    
    Returns a list of property dictionaries.
    """
    if not polygons:
        return []

    image_shape = full_hsv.shape[:2]
    
    # Use np.int32 for the mask to support many polygons (>255).
    polygon_id_mask = np.zeros(image_shape, dtype=np.int32)
    
    geo_properties = []
    valid_polygon_indices = []

    for i, polygon in enumerate(polygons):
        poly_id = i + 1
        if not polygon or len(polygon) < 3:
            # Add a placeholder for consistent list length
            geo_properties.append(None)
            continue

        poly_np = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(polygon_id_mask, [poly_np], poly_id)
        
        area = cv2.contourArea(poly_np)
        roundness = calculate_roundness(poly_np) # Assuming you have this helper
        x, y, w, h = cv2.boundingRect(poly_np)
        
        M = cv2.moments(poly_np)
        cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (x + w//2, y + h//2)

        geo_properties.append({'area': area, 'roundness': roundness, 'bbox': [x, y, x+w, y+h], 'centroid': (cx, cy)})
        valid_polygon_indices.append(poly_id)
        
    if not valid_polygon_indices:
        return []

    # --- FIX IS HERE ---
    # Calculate the mean for each channel separately, as the mask is 2D and the image is 3D.
    mean_h = ndimage.mean(full_hsv[:, :, 0], labels=polygon_id_mask, index=valid_polygon_indices)
    mean_s = ndimage.mean(full_hsv[:, :, 1], labels=polygon_id_mask, index=valid_polygon_indices)
    mean_v = ndimage.mean(full_hsv[:, :, 2], labels=polygon_id_mask, index=valid_polygon_indices)
    
    # Zip the results together into a list of (h, s, v) tuples
    mean_hsvs = list(zip(mean_h, mean_s, mean_v))
    # --- END OF FIX ---
    
    # Combine geometric and color properties
    all_properties = []
    for i, poly_id in enumerate(valid_polygon_indices):
        props = geo_properties[poly_id - 1] # Get pre-calculated geometry
        props['h_mean'], props['s_mean'], props['v_mean'] = mean_hsvs[i]
        all_properties.append(props)
        
    return all_properties

def parse_annotations_csv(text_or_bytes):
    logger.info(f"parse_annotations_csv: Starting, input type: {type(text_or_bytes)}")
    if isinstance(text_or_bytes, (bytes, bytearray)):
        text = text_or_bytes.decode("utf-8")
    else:
        text = str(text_or_bytes)
    df = pd.read_csv(io.StringIO(text))
    logger.info(f"parse_annotations_csv: Completed, rows: {len(df)}, columns: {list(df.columns)}")
    return df

def load_image_from_request():
    """Unified image loading from either file upload or S3 — memory-safe via temporary files."""
    logger.info("load_image_from_request: Starting image load")
    if 'image' in request.files:
        # Load from uploaded file using a temporary file instead of .read()
        logger.info("load_image_from_request: Loading from uploaded file")
        img_file = request.files['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
            img_file.save(tmp.name)
            tmp_path = tmp.name

        # Decode directly from disk (low memory)
        image = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        logger.info(f"load_image_from_request: Image loaded from file, shape: {image.shape if image is not None else 'None'}")

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return image

    elif 's3_object_key' in request.form:
            logger.info("load_image_from_request: Loading from S3")

            s3_object_key = request.form['s3_object_key']
            logger.info(f"load_image_from_request: S3 key: {s3_object_key}")

            from utilities.aws_utility import s3_client, S3_BUCKET_NAME
            import os

            # Extract file extension from S3 key (fallback to .img)
            _, ext = os.path.splitext(s3_object_key)
            suffix = ext if ext else ".img"

            # Stream S3 file directly to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                s3_client.download_fileobj(Bucket=S3_BUCKET_NAME, Key=s3_object_key, Fileobj=tmp)
                tmp_path = tmp.name

            # Decode directly from disk (OpenCV handles many formats natively)
            image = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
            logger.info(f"load_image_from_request: Image loaded from S3, shape: {image.shape if image is not None else 'None'}")

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

            return image

    logger.warning("load_image_from_request: No image source found")
    return None

def calculate_roundness(contour):
    """
    Calculate roundness of a contour (0-100, where 100 is perfectly circular)
    """
    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity: 4π × area / perimeter²
    # This gives 1.0 for a perfect circle, and lower values for irregular shapes
    if perimeter == 0:  # Avoid division by zero
        return 0
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Convert to percentage (0-100)
    roundness = min(circularity * 100, 100)  # Cap at 100
    return roundness

def extract_cell_properties_from_polygon(image, polygon, full_hsv=None):
    """
    Extract HSV, area, and roundness properties from a single cell polygon.

    Args:
        image: OpenCV image array
        polygon: List of (x,y) coordinate tuples
        full_hsv: Precomputed HSV image (optional, for speed)

    Returns:
        dict: Cell properties including HSV stats, area, roundness
    """
    if not polygon or len(polygon) < 3:
        return None

    # Convert polygon to numpy array
    poly_np = np.array(polygon, dtype=np.int32)

    # Create mask for this cell
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 255)

    # Extract cell region
    cell_region = cv2.bitwise_and(image, image, mask=mask)

    # Get bounding box of the polygon
    x, y, w, h = cv2.boundingRect(poly_np)

    # Extract the cell patch
    cell_patch = cell_region[y:y+h, x:x+w]
    if cell_patch.size == 0:
        return None

    # Convert to HSV (use precomputed if available)
    if full_hsv is not None:
        cell_hsv = full_hsv[y:y+h, x:x+w]
    else:
        cell_hsv = cv2.cvtColor(cell_patch, cv2.COLOR_BGR2HSV)

    # Calculate HSV statistics for the cell
    hsv_values = cell_hsv[mask[y:y+h, x:x+w] > 0]  # Only masked pixels

    if len(hsv_values) == 0:
        return None

    # Calculate area and roundness
    area = cv2.contourArea(poly_np)
    roundness = calculate_roundness(poly_np)

    # Calculate HSV statistics
    h_mean, h_std = np.mean(hsv_values[:, 0]), np.std(hsv_values[:, 0])
    s_mean, s_std = np.mean(hsv_values[:, 1]), np.std(hsv_values[:, 1])
    v_mean, v_std = np.mean(hsv_values[:, 2]), np.std(hsv_values[:, 2])

    return {
        'area': area,
        'roundness': roundness,
        'h_mean': h_mean,
        'h_std': h_std,
        's_mean': s_mean,
        's_std': s_std,
        'v_mean': v_mean,
        'v_std': v_std,
        'bbox': [x, y, x+w, y+h],
        'centroid': (x + w//2, y + h//2)
    }

def calculate_dynamic_parameters(selected_properties, strictness=5):
    """
    Simplified parameter calculation with linear strictness correlation.
    strictness: 1-10, where 1=most lenient, 10=most strict
    """
    logger.info(f"calculate_dynamic_parameters: Starting with {len(selected_properties) if selected_properties else 0} properties, strictness={strictness}")
    if not selected_properties:
        return {
            'h_min': 62, 'h_max': 133,
            's_min': 148, 's_max': 255,
            'v_min': 189, 'v_max': 255,
            'min_area': 718, 'max_area': 10000,
            'min_roundness': 46
        }

    # Extract property values
    # Extract property values - FULLY VECTORIZED with NumPy (no Python loops)
    selected_array = np.array([(p['area'], p['roundness'], p['h_mean'], p['s_mean'], p['v_mean']) 
                               for p in selected_properties], dtype=float)
    
    areas = selected_array[:, 0]
    roundness_values = selected_array[:, 1]
    h_values = selected_array[:, 2]
    s_values = selected_array[:, 3]
    v_values = selected_array[:, 4]

    # Calculate statistics
    h_mean, h_std = np.mean(h_values), np.std(h_values) if len(h_values) > 1 else 15.0
    s_mean, s_std = np.mean(s_values), np.std(s_values) if len(s_values) > 1 else 30.0
    v_mean, v_std = np.mean(v_values), np.std(v_values) if len(v_values) > 1 else 30.0
    avg_area = np.mean(areas)
    min_roundness = min(roundness_values)

    # Linear strictness scaling: higher strictness = narrower ranges, stricter thresholds
    # strictness 1 (lenient): factor=2.5, strictness 10 (strict): factor=0.5
    strictness_factor = 2.5 - (strictness - 1) * 0.2  # 2.5 (lenient) to 0.5 (strict)

    # HSV ranges: mean ± factor*std, clamped to valid ranges
    h_range = max(strictness_factor * h_std, 20)  # Minimum range of 20
    s_range = max(strictness_factor * s_std, 30)  # Minimum range of 30
    v_range = max(strictness_factor * v_std, 30)  # Minimum range of 30

    h_min = max(0, int(h_mean - h_range))
    h_max = min(179, int(h_mean + h_range))
    s_min = max(0, int(s_mean - s_range))
    s_max = min(255, int(s_mean + s_range))
    v_min = max(0, int(v_mean - v_range))
    v_max = min(255, int(v_mean + v_range))

    # Area range: scale with strictness
    area_tolerance = 0.3 + (strictness - 1) * 0.1  # 0.3 (lenient) to 1.2 (strict)
    min_area = max(50, int(avg_area * (1 - area_tolerance)))
    max_area = int(avg_area * (1 + area_tolerance * 2))  # Upper bound more flexible

    # Roundness threshold: stricter with higher strictness
    roundness_buffer = -20 + (strictness - 1) * 2  # -20 (lenient) to 0 (strict)
    min_roundness = max(10, int(min_roundness + roundness_buffer))

    params = {
        'h_min': h_min, 'h_max': h_max,
        's_min': s_min, 's_max': s_max,
        'v_min': v_min, 'v_max': v_max,
        'min_area': min_area, 'max_area': max_area,
        'min_roundness': min_roundness
    }
    logger.info(f"calculate_dynamic_parameters: Completed, params={params}")
    return params

def detect_lymphocytes_with_dynamic_params(image, dynamic_params):
    """
    Detect lymphocytes using dynamically calculated parameters.

    Args:
        image: OpenCV image array
        dynamic_params: Dictionary of detection parameters

    Returns:
        dict: Detection results with cells and metadata
    """
    # Extract parameters
    h_min = dynamic_params['h_min']
    h_max = dynamic_params['h_max']
    s_min = dynamic_params['s_min']
    s_max = dynamic_params['s_max']
    v_min = dynamic_params['v_min']
    v_max = dynamic_params['v_max']
    min_area = dynamic_params['min_area']
    max_area = dynamic_params['max_area']
    min_roundness = dynamic_params['min_roundness']

    # Store original image for reference
    original_img = image.copy()
    image_shape = image.shape[:2]  # (height, width)

    # Downscale original image for processing (preserve original for display)
    process_scale = 1.0  # Default: no scaling
    # Only downscale large images to speed up processing
    if max(image.shape[:2]) > 1000:
        process_scale = 0.5  # Process at 50% resolution
        image = cv2.resize(image, (0, 0), fx=process_scale, fy=process_scale,
                         interpolation=cv2.INTER_AREA)

    # Convert to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Also prepare HSV for the original (full-resolution) image so HSV stats
    # for detected contours can be computed in the same color space as selected examples.
    try:
        orig_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    except Exception:
        orig_hsv = img_hsv

    # Create mask with dynamic HSV values
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    hsv_mask = cv2.inRange(img_hsv, lower, upper)

    # Apply morphological operations to clean up mask
    kernel = np.ones((5,5), np.uint8)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours from the HSV mask
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Adjust area constraints for the scale if we're using a downsampled image
    scaled_min_area = min_area * (process_scale ** 2) if process_scale != 1 else min_area
    scaled_max_area = max_area * (process_scale ** 2) if process_scale != 1 else max_area

    # Lists to store detected cells
    detected_cells = []

    # Process each contour - first filter by area
    area_filtered_contours = [
        cnt for cnt in contours
        if scaled_min_area < cv2.contourArea(cnt) < scaled_max_area
    ]

    # Filter and process by roundness
    for contour in area_filtered_contours:
        # Calculate roundness
        roundness = calculate_roundness(contour)

        # Only keep contours that meet roundness criteria
        if roundness >= min_roundness:
            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2

            # Scale coordinates back to original image size if we downsampled
            if process_scale != 1:
                cx = int(cx / process_scale)
                cy = int(cy / process_scale)
                x = int(x / process_scale)
                y = int(y / process_scale)
                w = int(w / process_scale)
                h = int(h / process_scale)
                area = int(area / (process_scale ** 2))

            # Store cell information
            cell_info = {
                'x': float(cx),
                'y': float(cy),
                'area': int(area),
                'roundness': float(roundness),
                'bbox': [int(x), int(y), int(x + w), int(y + h)]
            }
            # Compute mean HSV for the contour region. If we downscaled the image for processing,
            # map the contour back to original image coordinates and sample from orig_hsv so
            # HSV stats are comparable to selected examples (which were extracted at full res).
            try:
                if process_scale != 1.0:
                    # scale contour points back to original image coordinates
                    scale_factor = 1.0 / process_scale
                    scaled_cnt = (contour.astype(np.float32) * scale_factor).astype(np.int32)
                    mask = np.zeros(orig_hsv.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [scaled_cnt], -1, 255, thickness=-1)
                    hsv_region = orig_hsv[mask > 0]
                else:
                    mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                    hsv_region = img_hsv[mask > 0]

                if hsv_region.size > 0:
                    h_m = float(np.mean(hsv_region[:, 0]))
                    s_m = float(np.mean(hsv_region[:, 1]))
                    v_m = float(np.mean(hsv_region[:, 2]))
                else:
                    h_m, s_m, v_m = 0.0, 0.0, 0.0
            except Exception:
                h_m, s_m, v_m = 0.0, 0.0, 0.0

            # Attach HSV means (these now refer to full-resolution HSV space when possible)
            cell_info.update({'h_mean': h_m, 's_mean': s_m, 'v_mean': v_m})
            detected_cells.append(cell_info)

    # Prepare result
    result = {
        'cells': detected_cells,
        'count': len(detected_cells),
        'image_shape': image_shape,
        'parameters': dynamic_params
    }

# ----------------------
# New dynamic cell detection endpoint
# ----------------------

@dynamic_cells_route_bp.route("/detect-from-selected", methods=["POST"])
@protected
def detect_from_selected_endpoint(decoded_token):
    """
    Dynamic cell detection endpoint that calculates detection parameters from selected cells
    and finds similar cells in the image.

    Expected input:
    - image: uploaded image file OR s3_object_key: S3 object key
    - jobId: job ID to load cell predictions CSV
    - annotations: CSV with selected cells (x, y, selected columns)

    Returns: Detected cells with dynamic parameters and similarity scores
    """
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")

    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="dynamic_cell_detection",
        status="start"
    )

    try:
        # Load image using unified function
        logger.info("detect_from_selected_endpoint: Loading image")
        image = load_image_from_request()
        if image is None:
            return jsonify({"error": "Missing 'image' file or 's3_object_key' parameter"}), 400

        # Get job_id to load cell predictions CSV
        job_id = request.form.get('jobId')
        logger.info(f"detect_from_selected_endpoint: Job ID: {job_id}")
        if not job_id:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="dynamic_cell_detection",
                status="error",
                metadata={"reason": "missing_job_id"}
            )
            return jsonify({"error": "Missing jobId parameter"}), 400

        # Load full cell predictions CSV from database/S3
        logger.info(f"detect_from_selected_endpoint: Querying database for job_id: {job_id}")
        sample = Sample.query.filter_by(job_id=job_id).first()
        if not sample:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="dynamic_cell_detection",
                status="error",
                metadata={"reason": "sample_not_found", "job_id": job_id}
            )
            return jsonify({"error": f"Sample with job_id {job_id} not found"}), 404

        # Load the full CSV with polygons
        logger.info(f"detect_from_selected_endpoint: Loading CSV from S3")
        try:
            import boto3
            from utilities.aws_utility import s3_client, S3_BUCKET_NAME

            logger.info(f"detect_from_selected_endpoint: S3 key: {sample.s3_inference_key}")
            s3_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=sample.s3_inference_key)
            csv_data = s3_response['Body'].read().decode('utf-8')
            logger.info(f"CSV data length: {len(csv_data)}")
            full_df = pd.read_csv(io.StringIO(csv_data))
            logger.info(f"CSV columns: {list(full_df.columns)}")
            logger.info(f"CSV shape: {full_df.shape}")

            # Parse polygons and create polygons list
            logger.info(f"detect_from_selected_endpoint: Parsing polygons from CSV")
            polygons_list = []
            for idx, row in full_df.iterrows():
                poly_x = []
                poly_y = []
                if pd.notna(row.get('poly_x')) and pd.notna(row.get('poly_y')):
                    try:
                        poly_x = [float(x.strip()) for x in str(row['poly_x']).split(',') if x.strip()]
                        poly_y = [float(y.strip()) for y in str(row['poly_y']).split(',') if y.strip()]
                        polygons_list.append(list(zip(poly_x, poly_y)))
                    except Exception as poly_error:
                        logger.info(f"Error parsing polygon for row {idx}: {poly_error}")
                        polygons_list.append([])
                else:
                    polygons_list.append([])

            logger.info(f"detect_from_selected_endpoint: Parsed {len(polygons_list)} polygons successfully")

        except Exception as csv_error:
            logger.info(f"CSV loading error: {csv_error}")
            import traceback
            traceback.print_exc()
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="dynamic_cell_detection",
                status="error",
                metadata={"reason": "csv_load_failed", "job_id": job_id, "error": str(csv_error)}
            )
            return jsonify({"error": f"Failed to load CSV for job_id {job_id}: {str(csv_error)}"}), 400

        # Get selected examples from annotations CSV
        strictness = int(request.form.get('strictness', 5))  # Default to 5 (medium strictness)
        logger.info(f"detect_from_selected_endpoint: Strictness level: {strictness}")
        if 'annotations' not in request.form and 'annotations' not in request.files:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="dynamic_cell_detection",
                status="error",
                metadata={"reason": "no_annotations_provided"}
            )
            return jsonify({"error": "No annotations CSV provided"}), 400

        if 'annotations' in request.form:
            annotations_csv = request.form['annotations']
        else:
            annotations_csv = request.files['annotations'].read().decode('utf-8')

        logger.info(f"detect_from_selected_endpoint: Parsing annotations CSV")
        annotations_df = parse_annotations_csv(annotations_csv)
        if not {'x', 'y', 'selected'}.issubset(annotations_df.columns):
            return jsonify({"error": "annotations CSV must contain 'x', 'y', and 'selected' columns"}), 400

        # Find selected cells by matching centroids
        logger.info(f"detect_from_selected_endpoint: Matching selected cells to polygons")
        selected_indices = []
        selected_polygons = []
        for idx, row in annotations_df.iterrows():
            if int(row['selected']) == 1:
                ex_x = float(row['x'])
                ex_y = float(row['y'])

                # Find matching cell in full CSV by centroid proximity
                best_i = None
                best_dist = float('inf')
                for i, full_row in full_df.iterrows():
                    if pd.notna(full_row.get('x0')) and pd.notna(full_row.get('y0')) and pd.notna(full_row.get('x1')) and pd.notna(full_row.get('y1')):
                        # Calculate centroid of bounding box
                        cx = (float(full_row['x0']) + float(full_row['x1'])) / 2
                        cy = (float(full_row['y0']) + float(full_row['y1'])) / 2

                        dist = (cx - ex_x)**2 + (cy - ex_y)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_i = i

                if best_i is not None and best_dist < (20**2):  # Within 20 pixels
                    selected_indices.append(best_i)
                    if best_i < len(polygons_list):
                        selected_polygons.append(polygons_list[best_i])

        logger.info(f"detect_from_selected_endpoint: Found {len(selected_indices)} selected examples")
        
        # Clean up annotations dataframe - no longer needed
        del annotations_df
        del annotations_csv
        
        if not selected_indices:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="dynamic_cell_detection",
                status="error",
                metadata={"reason": "no_selected_examples_found"}
            )
            return jsonify({"error": "No selected examples could be matched to cell predictions"}), 400

        logger.info(f"detect_from_selected_endpoint: Matched {len(selected_polygons)} selected cell polygons")
        
        # Clean up full_df - no longer needed after matching
        del full_df

        # VECTORIZED PROPERTY EXTRACTION FOR SELECTED CELLS
        logger.info(f"detect_from_selected_endpoint: Extracting properties from selected cells")
        selected_properties = _calculate_properties_iterative(selected_polygons, image)
        # height, width, channels = image.shape
        # num_pixels = height * width
        # if num_pixels > LARGE_IMAGE_PIXEL_THRESHOLD:
        #         logger.warning(f"Image is large ({num_pixels} pixels), using memory-safe iterative method.")
        #         # Call the safe, one-by-one method
        #         selected_properties = _calculate_properties_iterative(selected_polygons, image)
        # else:
        #     logger.info(f"detect_from_selected_endpoint: Converting image to HSV (shape: {image.shape})")
        #     full_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #     logger.info(f"detect_from_selected_endpoint: HSV conversion complete")
        #     selected_properties = _calculate_polygon_properties_vectorized(selected_polygons,full_hsv)

        logger.info(f"detect_from_selected_endpoint: Successfully extracted properties from {len(selected_properties)} selected cells")
        
        # Clean up selected polygons - no longer needed
        del selected_polygons

        # Calculate dynamic detection parameters
        logger.info(f"detect_from_selected_endpoint: Calculating dynamic parameters")
        dynamic_params = calculate_dynamic_parameters(selected_properties, strictness)
        logger.info(f"detect_from_selected_endpoint: Dynamic params - H({dynamic_params['h_min']}-{dynamic_params['h_max']}) S({dynamic_params['s_min']}-{dynamic_params['s_max']}) V({dynamic_params['v_min']}-{dynamic_params['v_max']}) Area({dynamic_params['min_area']}-{dynamic_params['max_area']}) Roundness({dynamic_params['min_roundness']})")

        # Use the same full_hsv for candidates filtering

        # Use CSV polygons as candidates - OPTIMIZED VERSION
        logger.info(f"detect_from_selected_endpoint: Preparing selected centroids for filtering")
        selected_centroids = [tuple(map(float, p['centroid'])) for p in selected_properties if 'centroid' in p]

        # STEP 1: Fast pre-filtering by area and roundness only (no HSV yet)
        logger.info(f"detect_from_selected_endpoint: STEP 1 - Pre-filtering {len(polygons_list)} polygons by area/roundness")
        prefiltered_polygons = []
        for i, polygon in enumerate(polygons_list):
            if i in selected_indices or not polygon or len(polygon) < 3:
                continue

            poly_np = np.array(polygon, dtype=np.int32)
            area = cv2.contourArea(poly_np)
            if not (dynamic_params['min_area'] <= area <= dynamic_params['max_area']):
                continue

            roundness = calculate_roundness(poly_np)
            if roundness >= dynamic_params['min_roundness']:
                prefiltered_polygons.append((i, polygon, area, roundness))

        logger.info(f"detect_from_selected_endpoint: STEP 1 complete - Prefiltered to {len(prefiltered_polygons)} polygons")
        
        # Clean up polygons_list - no longer needed after pre-filtering
        del polygons_list

        # STEP 2: VECTORIZED HSV filtering for maximum speed
        logger.info(f"detect_from_selected_endpoint: STEP 2 - Vectorized HSV filtering for {len(prefiltered_polygons)} polygons")
        
        candidates = []
        if prefiltered_polygons:
            # Extract just the polygon coordinates
            prefiltered_coords = [polygon for _, polygon, _, _ in prefiltered_polygons]
            candidate_properties = _calculate_properties_iterative(prefiltered_coords, image)

            # Now filter the results in a simple Python loop (very fast, low memory)
            for props in candidate_properties:
                # HSV filtering
                h_ok = dynamic_params['h_min'] <= props['h_mean'] <= dynamic_params['h_max']
                s_ok = dynamic_params['s_min'] <= props['s_mean'] <= dynamic_params['s_max']
                v_ok = dynamic_params['v_min'] <= props['v_mean'] <= dynamic_params['v_max']
                
                if not (h_ok and s_ok and v_ok):
                    continue

                # Distance filtering - exclude cells too close to selected ones
                cx, cy = props['centroid']
                too_close = any((sx - cx)**2 + (sy - cy)**2 < 25 for sx, sy in selected_centroids)

                if not too_close:
                    candidates.append(props)


        # Clean up prefiltered polygons and full_hsv - no longer needed
        del prefiltered_polygons
        logger.info(f"detect_from_selected_endpoint: Cleaned up HSV image and prefiltered data")
        
        logger.info(f"detect_from_selected_endpoint: STEP 2 complete - Final candidates: {len(candidates)}")

        # Calculate target based on strictness: more lenient = more cells
        logger.info(f"detect_from_selected_endpoint: Calculating target suggestion count")
        base_target = min(200, len(selected_properties) * 15)
        target = min(MAX_RESULTS, int(base_target * (11 - strictness) / 5.0))  # Linear scaling
        target = max(target, 20)  # Minimum 20 cells
        logger.info(f"detect_from_selected_endpoint: Target suggestions: {target}")

        logger.info(f"detect_from_selected_endpoint: Creating suggestions list")
        suggestions_list = [
            {
                'label': f'csv_cell_{i}',
                'x': props['centroid'][0],
                'y': props['centroid'][1],
                'score': 1.0,
                'area': props['area'],
                'roundness': props['roundness'],
                'bbox': props['bbox']
            } for i, props in enumerate(candidates[:target])
        ]
        logger.info(f"detect_from_selected_endpoint: Created suggestions list with {len(suggestions_list)} items")
        
        # Clean up candidates list - no longer needed
        
        del selected_centroids

        selection_debug = {
            'method': 'csv_filter',
            'candidates_total': len(candidates),
            'chosen': len(suggestions_list),
            'target': target,
            'exclusion_radius': 5  # Fixed 5px exclusion radius
        }
        logger.info(f"detect_from_selected_endpoint: Final suggestions list: {len(suggestions_list)} cells")

        # Create CSV response
        logger.info(f"detect_from_selected_endpoint: Creating CSV response")
        if suggestions_list:
            suggestions_df = pd.DataFrame(suggestions_list)
            csv_buf = suggestions_df.to_csv(index=False)
            del suggestions_df  # Clean up dataframe
        else:
            csv_buf = "label,x,y,score,area,roundness,bbox\n"

        # Prepare response
        logger.info(f"detect_from_selected_endpoint: Preparing response payload")
        response_payload = {
            "csv": csv_buf,
            "num_candidates": len(candidates),
            "num_selected_examples": len(selected_indices),
            "num_suggestions": len(suggestions_list),
            "estimated_error_rate": 0.40,  # ±40% tolerance
            "dynamic_parameters": dynamic_params,
            "debug": {
                "detection_mode": "csv_polygon_filter",
                "tolerance": "dynamic params",
                "selected_properties_count": len(selected_properties),
                "detected_cells_count": len(candidates),
                "image_shape": image.shape[:2]
            },
            "selection_debug": selection_debug if 'selection_debug' in locals() else {},
            "suggestions_list": suggestions_list
        }

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="dynamic_cell_detection",
            status="success",
            metadata={
                "selected_examples_count": len(selected_indices),
                "detected_cells_count": len(candidates),
                "suggestions_count": len(suggestions_list),
                "dynamic_params": dynamic_params
            }
        )

        logger.info(f"detect_from_selected_endpoint: Request completed successfully")
        
        # Final cleanup before returning
        del image
        del selected_properties
        del selected_indices
        logger.info(f"detect_from_selected_endpoint: Final memory cleanup complete")
        
        return jsonify(response_payload)

    except Exception as e:
        logger.info(f"Dynamic cell detection error: {e}")
        import traceback
        traceback.print_exc()
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="dynamic_cell_detection",
            status="error",
            metadata={"error": str(e)}
        )
        return jsonify({"error": "internal_server_error", "message": str(e)}), 500
