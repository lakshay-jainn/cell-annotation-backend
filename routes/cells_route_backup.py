# cells_route_auto.py
import io
import traceback
import math
import json

from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
import cv2
from skimage import measure
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler, normalize

from utilities.auth_utility import protected
from utilities.logging_utility import ActivityLogger

cells_route_bp = Blueprint("cells", __name__)

# PARAMETERS you can tweak
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
INT_HIST_BINS = 16

# Selection thresholds / weights (tunable)
SIMILARITY_THRESHOLD = 0.55   # final score threshold to include a cell
MAX_RESULTS = 500

WEIGHT_MAHAL = 0.50
WEIGHT_COS = 0.20
WEIGHT_COLOR = 0.25
WEIGHT_SIZE = 0.05

# ----------------------
# Helpers (segmentation / feature extraction re-used from your file)
# ----------------------

def parse_annotations_csv(text_or_bytes):
    if isinstance(text_or_bytes, (bytes, bytearray)):
        text = text_or_bytes.decode("utf-8")
    else:
        text = str(text_or_bytes)
    df = pd.read_csv(io.StringIO(text))
    return df

def load_image_from_file_storage(f):
    in_memory = f.read()
    arr = np.frombuffer(in_memory, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def segment_cells(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        if np.mean(g[th == 255]) < np.mean(g[th == 0]):
            th = cv2.bitwise_not(th)
    except Exception:
        pass
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    labeled = measure.label(th // 255, connectivity=2)
    props = list(measure.regionprops(labeled, intensity_image=gray))
    return labeled, props, th

# extract_region_features and extract_features_from_patch copied/adapted
def extract_region_features(region, gray_image, full_color_image):
    features = {}
    features['area'] = float(region.area)
    features['perimeter'] = float(region.perimeter) if region.perimeter is not None else 0.0
    features['solidity'] = float(region.solidity) if hasattr(region, 'solidity') else 0.0
    features['eccentricity'] = float(region.eccentricity) if hasattr(region, 'eccentricity') else 0.0
    features['major_axis_length'] = float(region.major_axis_length) if hasattr(region, 'major_axis_length') else 0.0
    features['minor_axis_length'] = float(region.minor_axis_length) if hasattr(region, 'minor_axis_length') else 0.0
    if region.perimeter and region.perimeter > 0:
        features['roundness'] = 4.0 * np.pi * region.area / (region.perimeter ** 2)
    else:
        features['roundness'] = 0.0

    mask = region.filled_image.astype(np.uint8)
    mask255 = (mask * 255).astype(np.uint8)
    try:
        hu = cv2.HuMoments(cv2.moments(mask255)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-16)
    except Exception:
        hu = np.zeros(7)
    for i, val in enumerate(hu[:7]):
        features[f'hu_{i+1}'] = float(val)

    minr, minc, maxr, maxc = region.bbox
    patch = gray_image[minr:maxr, minc:maxc]
    try:
        hist = cv2.calcHist([patch], [0], mask255, [INT_HIST_BINS], [0,256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
    except Exception:
        hist = np.zeros(INT_HIST_BINS)
    for i, v in enumerate(hist):
        features[f'int_hist_{i}'] = float(v)

    try:
        lbp = local_binary_pattern(patch, LBP_N_POINTS, LBP_RADIUS, method="uniform")
        lbp_masked = lbp[mask == 1]
        n_bins = LBP_N_POINTS + 2
        if lbp_masked.size > 0:
            lbp_hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
        else:
            lbp_hist = np.zeros(n_bins)
    except Exception:
        lbp_hist = np.zeros(LBP_N_POINTS + 2)
    for i, v in enumerate(lbp_hist):
        features[f'lbp_{i}'] = float(v)

    # color features
    h_mean = s_mean = v_mean = 0.0
    try:
        if full_color_image is not None:
            color_patch = full_color_image[minr:maxr, minc:maxc]
            if color_patch.size != 0 and mask.size != 0:
                hsv_patch = cv2.cvtColor(color_patch, cv2.COLOR_BGR2HSV)
                mask_bool = mask.astype(bool)
                if mask_bool.any():
                    h_vals = hsv_patch[..., 0][mask_bool].astype(np.float32)
                    s_vals = hsv_patch[..., 1][mask_bool].astype(np.float32)
                    v_vals = hsv_patch[..., 2][mask_bool].astype(np.float32)
                    h_rad = (h_vals / 180.0) * 2.0 * math.pi
                    hx = np.cos(h_rad).mean() if h_rad.size else 0.0
                    hy = np.sin(h_rad).mean() if h_rad.size else 0.0
                    if hx == 0 and hy == 0:
                        h_mean = float(h_vals.mean()) if h_vals.size else 0.0
                    else:
                        h_mean_deg = (math.atan2(hy, hx) / (2.0 * math.pi)) * 180.0
                        if h_mean_deg < 0:
                            h_mean_deg += 180.0
                        h_mean = float(h_mean_deg)
                    s_mean = float(s_vals.mean()) if s_vals.size else 0.0
                    v_mean = float(v_vals.mean()) if v_vals.size else 0.0
    except Exception:
        h_mean = s_mean = v_mean = 0.0

    features['h_mean'] = float(h_mean)
    features['s_mean'] = float(s_mean)
    features['v_mean'] = float(v_mean)

    cy, cx = region.centroid
    features['centroid_x'] = float(cx)
    features['centroid_y'] = float(cy)
    return features

def extract_features_from_patch(cx, cy, gray, patch_radius=16):
    h, w = gray.shape
    x1 = int(max(0, cx - patch_radius))
    y1 = int(max(0, cy - patch_radius))
    x2 = int(min(w, cx + patch_radius))
    y2 = int(min(h, cy + patch_radius))
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        class EmptyRegion:
            def __init__(self, bbox, centroid, patch):
                self.filled_image = np.ones_like(patch, dtype=np.uint8)
                self.area = patch.size
                self.perimeter = float(2 * (patch.shape[0] + patch.shape[1]))
                self.solidity = 1.0
                self.eccentricity = 0.0
                self.major_axis_length = max(patch.shape) if patch.size else 0.0
                self.minor_axis_length = min(patch.shape) if patch.size else 0.0
                self.bbox = bbox
                self.centroid = centroid
        bbox = (y1, x1, y2, x2)
        centroid = (cy, cx)
        region = EmptyRegion(bbox, centroid, patch)
        return extract_region_features(region, gray, None)
    try:
        _, th = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        th = (patch > patch.mean()).astype(np.uint8) * 255
    labeled = measure.label(th // 255, connectivity=2)
    props = list(measure.regionprops(labeled, intensity_image=patch))
    if len(props) == 0:
        class SimpleRegion:
            def __init__(self, bbox, centroid, patch):
                self.filled_image = np.ones_like(patch, dtype=np.uint8)
                self.area = patch.size
                self.perimeter = float(2 * (patch.shape[0] + patch.shape[1]))
                self.solidity = 1.0
                self.eccentricity = 0.0
                self.major_axis_length = max(patch.shape)
                self.minor_axis_length = min(patch.shape)
                self.bbox = bbox
                self.centroid = centroid
        bbox = (y1, x1, y2, x2)
        centroid = (cy, cx)
        region = SimpleRegion(bbox, centroid, patch)
        return extract_region_features(region, gray, None)
    else:
        props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
        p = props_sorted[0]
        class SimpleRegionFromProp:
            def __init__(self, prop, offset_y, offset_x):
                self.filled_image = prop.filled_image.astype(np.uint8)
                self.area = getattr(prop, 'area', int(self.filled_image.sum()))
                self.perimeter = float(getattr(prop, 'perimeter', 0.0))
                self.solidity = float(getattr(prop, 'solidity', 0.0))
                self.eccentricity = float(getattr(prop, 'eccentricity', 0.0))
                self.major_axis_length = float(getattr(prop, 'major_axis_length', 0.0))
                self.minor_axis_length = float(getattr(prop, 'minor_axis_length', 0.0))
                minr, minc, maxr, maxc = prop.bbox
                self.bbox = (minr + offset_y, minc + offset_x, maxr + offset_y, maxc + offset_x)
                rr, cc = prop.centroid
                self.centroid = (rr + offset_y, cc + offset_x)
        region = SimpleRegionFromProp(p, y1, x1)
        return extract_region_features(region, gray, None)

# ----------------------
# fast_detect_centroids (adapted)
# ----------------------
def fast_detect_centroids(gray: np.ndarray, downscale: float = 1.0, dt_peak_rel: float = 0.25, merge_radius: float = 6.0):
    if downscale != 1.0:
        h, w = gray.shape
        gs = cv2.resize(gray, (int(w*downscale), int(h*downscale)), interpolation=cv2.INTER_AREA)
    else:
        gs = gray.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gs_proc = clahe.apply(gs)
    mask = cv2.adaptiveThreshold(gs_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 7)
    mker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, mker, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mker, iterations=1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    maxv = dist.max() if dist.max() > 0 else 1.0
    thresh = dt_peak_rel * maxv
    dil = cv2.dilate(dist, np.ones((3,3), np.uint8))
    local_max = (dist == dil) & (dist >= thresh)
    ys, xs = np.nonzero(local_max)
    coords = []
    for y, x in zip(ys, xs):
        if downscale != 1.0:
            xf = float(x) / downscale
            yf = float(y) / downscale
        else:
            xf = float(x); yf = float(y)
        coords.append((xf, yf, 'dt'))
    merged = []
    for c in coords:
        x,y,src = c
        too_close = False
        for m in merged:
            if math.hypot(m[0]-x, m[1]-y) <= merge_radius:
                too_close = True; break
        if not too_close:
            merged.append((x,y,src))
    return merged, mask if downscale==1.0 else cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

# ----------------------
# New endpoints: /auto_detect and /auto_select
# ----------------------

@cells_route_bp.route("/auto_detect", methods=["POST"])
@protected
def auto_detect_endpoint(decoded_token):
    """
    Accepts an uploaded image file (field 'image') and optional form param 'downscale'.
    Returns: {'candidates': [{'x','y','src'}], 'count': int}
    """
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="auto_detect_cells",
        status="start"
    )
    
    try:
        if 'image' not in request.files:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="auto_detect_cells",
                status="error",
                metadata={"reason": "missing_image_file"}
            )
            return jsonify({"error": "Missing 'image' file"}), 400
        img_file = request.files['image']
        image = load_image_from_file_storage(img_file)
        if image is None:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="auto_detect_cells",
                status="error",
                metadata={"reason": "could_not_decode_image"}
            )
            return jsonify({"error": "Could not decode image"}), 400
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        downscale = float(request.form.get('downscale', 0.6))
        candidates, mask = fast_detect_centroids(gray, downscale=downscale, dt_peak_rel=0.22, merge_radius=6.0)
        res = [{'x': float(x), 'y': float(y), 'src': src} for x,y,src in candidates]
        
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="auto_detect_cells",
            status="success",
            metadata={"candidates_count": len(res), "downscale": downscale}
        )
        
        return jsonify({'candidates': res, 'count': len(res)})
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="auto_detect_cells",
            status="error",
            metadata={"error": str(e)}
        )
        traceback.print_exc()
        return jsonify({"error":"internal_server_error", "message": str(e)}), 500

@cells_route_bp.route("/auto-select-similar", methods=["POST"])
@protected
def auto_select_endpoint(decoded_token):
    user_id = decoded_token.get("user_id")
    user_role = decoded_token.get("role")
    
    ActivityLogger.log_activity(
        user_id=user_id,
        user_role=user_role,
        action_type="api_call",
        action_details="auto_select_similar_cells",
        status="start"
    )

    try:
        # Load image - either from uploaded file or S3 object key
        image = None

        if 'image' in request.files:
            # Load from uploaded file (legacy support)
            img_file = request.files['image']
            image = load_image_from_file_storage(img_file)
        elif 's3_object_key' in request.form:
            # Load directly from S3 using object key
            s3_object_key = request.form['s3_object_key']
            try:
                import boto3
                from utilities.aws_utility import s3_client, S3_BUCKET_NAME

                # Download image from S3
                s3_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_object_key)
                image_data = s3_response['Body'].read()

                # Convert to numpy array
                import numpy as np
                from PIL import Image
                import io

                pil_image = Image.open(io.BytesIO(image_data))
                image = np.array(pil_image)

                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]  # RGB to BGR

            except Exception as s3_error:
                ActivityLogger.log_activity(
                    user_id=user_id,
                    user_role=user_role,
                    action_type="api_call",
                    action_details="auto_select_similar_cells",
                    status="error",
                    metadata={"reason": "s3_load_failed", "s3_key": s3_object_key, "error": str(s3_error)}
                )
                return jsonify({"error": f"Failed to load image from S3: {str(s3_error)}"}), 400
        else:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="auto_select_similar_cells",
                status="error",
                metadata={"reason": "missing_image_source"}
            )
            return jsonify({"error": "Missing 'image' file or 's3_object_key' parameter"}), 400

        if image is None:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="auto_select_similar_cells",
                status="error",
                metadata={"reason": "could_not_decode_image"}
            )
            return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get annotation/examples: priority to CSV in form/file, else JSON 'examples'
        examples_df = None
        if 'annotations' in request.form or 'annotations' in request.files:
            if 'annotations' in request.form:
                annotations_csv = request.form['annotations']
            else:
                annotations_csv = request.files['annotations'].read().decode('utf-8')
            examples_df = parse_annotations_csv(annotations_csv)
            if not {'x','y'}.issubset(examples_df.columns):
                return jsonify({"error":"annotations CSV must contain 'x' and 'y' columns"}), 400
        else:
            # try JSON body or form field 'examples'
            examples_field = None
            if request.is_json:
                examples_field = request.get_json(force=True).get('examples', None)
            if examples_field is None:
                examples_field = request.form.get('examples', None)
            if examples_field is None:
                ActivityLogger.log_activity(
                    user_id=user_id,
                    user_role=user_role,
                    action_type="api_call",
                    action_details="auto_select_similar_cells",
                    status="error",
                    metadata={"reason": "no_examples_provided"}
                )
                return jsonify({"error":"No annotations/examples provided"}), 400
            # parse if string
            if isinstance(examples_field, str):
                try:
                    examples_list = json.loads(examples_field)
                except Exception:
                    return jsonify({"error":"Failed to parse 'examples' JSON"}), 400
            else:
                examples_list = examples_field
            # build DataFrame
            df = pd.DataFrame(examples_list)
            if not {'x','y'}.issubset(df.columns):
                return jsonify({"error":"examples JSON must contain x and y"}), 400
            # ensure 'selected' column exists (0/1). default 1 for provided examples? prefer explicit.
            if 'selected' not in df.columns:
                df['selected'] = 1
            examples_df = df

        # Segment candidates across whole image
        labeled, props, mask_binary = segment_cells(gray)

        # Compute features for each region (props)
        region_features = []    
        for idx, p in enumerate(props):
            feats = extract_region_features(p, gray, image)
            feats['label'] = f"region_{len(region_features)}"
            feats['orig_region_index'] = idx   # <<-- add this
            region_features.append(feats)

        # Build candidate DataFrame
        cand_df = pd.DataFrame(region_features)
        if cand_df.empty:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="auto_select_similar_cells",
                status="success",
                metadata={"candidates_count": 0, "examples_count": len(examples_df), "suggestions_count": 0}
            )
            return jsonify({"suggestions": [], "num_candidates": 0, "num_selected_examples": 0, "num_suggestions": 0})

        # Map provided example points to candidate rows: find nearest candidate centroid within a threshold
        mapped_candidates = []
        sel_flags = []
        for idx, row in examples_df.iterrows():
            ex_x = float(row['x'])
            ex_y = float(row['y'])
            selected_flag = int(row.get('selected', 1))
            best_i = None
            best_d = float('inf')
            for i, rf in enumerate(region_features):
                dx = rf['centroid_x'] - ex_x
                dy = rf['centroid_y'] - ex_y
                d = dx*dx + dy*dy
                if d < best_d:
                    best_d = d
                    best_i = i
            # if close enough (distance threshold), mark candidate as example; else extract patch features and append as external example
            if best_i is not None and best_d < (50**2):
                # mark this candidate row as example selected or not
                sel_flags.append((best_i, selected_flag))
            else:
                # get patch features and append as an extra pseudo-candidate (so examples are included in feature scaling)
                patch_feats = extract_features_from_patch(ex_x, ex_y, gray)
                patch_feats['label'] = row.get('label', f"example_{idx}")
                patch_feats['centroid_x'] = ex_x
                patch_feats['centroid_y'] = ex_y
                patch_feats['is_external_example'] = 1
                region_features.append(patch_feats)
                sel_flags.append((len(region_features)-1, selected_flag))

        # rebuild cand_df in case we appended
        cand_df = pd.DataFrame(region_features).fillna(0.0)

        # Build feature matrix: drop non-feature columns
        non_feature_cols = set(['label','centroid_x','centroid_y','is_external_example'])
        feature_cols = [c for c in cand_df.columns if c not in non_feature_cols]
        X = cand_df[feature_cols].astype(float).values

        # Standardize features
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Determine selected indices
        sel_idx = [i for i,flag in sel_flags if int(flag)==1] if len(sel_flags)>0 else []
        sel_idx = list(sorted(set(sel_idx)))

        if len(sel_idx) == 0:
            ActivityLogger.log_activity(
                user_id=user_id,
                user_role=user_role,
                action_type="api_call",
                action_details="auto_select_similar_cells",
                status="error",
                metadata={"reason": "no_selected_examples"}
            )
            return jsonify({"error":"No selected examples found in provided examples/annotations"}), 400

        # If too few examples for covariance, use global covariance fallback
        if len(sel_idx) >= 2:
            cov = np.cov(Xs[sel_idx,:], rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.pinv(cov)
            mean_sel = Xs[sel_idx,:].mean(axis=0)
        else:
            cov = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
            inv_cov = np.linalg.pinv(cov)
            mean_sel = Xs[sel_idx,:].mean(axis=0)

        # Mahalanobis distances and mapping to similarity
        dists = []
        for i in range(Xs.shape[0]):
            try:
                d = mahalanobis(Xs[i,:], mean_sel, inv_cov)
            except Exception:
                d = np.linalg.norm(Xs[i,:] - mean_sel)
            dists.append(d)
        dists = np.array(dists)
        d_norm = (dists - dists.mean()) / (dists.std() + 1e-9)
        sim_from_mahal = np.exp(- (d_norm ** 2) / 2.0)

        # Cosine similarity
        mean_vec = mean_sel.reshape(1, -1)
        X_normed = normalize(Xs)
        mean_normed = normalize(mean_vec)[0]
        cos_sim = (X_normed @ mean_normed).flatten()

        # Color similarity (if columns exist)
        color_sim = np.zeros(X.shape[0])
        if all(c in cand_df.columns for c in ['h_mean','s_mean','v_mean']):
            try:
                hsv_all = cand_df[['h_mean','s_mean','v_mean']].fillna(0.0).values.astype(float)
                sel_hsv = hsv_all[sel_idx, :] if len(sel_idx)>0 else hsv_all
                sel_hsv_mean = sel_hsv.mean(axis=0)
                def hue_dist(a, b):
                    da = abs(a - b)
                    da = min(da, 180.0 - da)
                    return da / 90.0
                cs = []
                for row in hsv_all:
                    dh = hue_dist(row[0], sel_hsv_mean[0])
                    ds = abs(row[1] - sel_hsv_mean[1]) / 255.0
                    dv = abs(row[2] - sel_hsv_mean[2]) / 255.0
                    score = math.exp(- (dh**2 + ds**2 + dv**2))
                    cs.append(score)
                color_sim = np.array(cs)
            except Exception:
                color_sim = np.zeros(X.shape[0])

        # Size similarity (area)
        size_sim = np.ones(X.shape[0])
        if 'area' in feature_cols:
            try:
                area_idx = feature_cols.index('area')
                areas = X[:, area_idx].astype(float)
                sel_areas = areas[sel_idx] if len(sel_idx)>0 else areas
                mean_area = float(sel_areas.mean()) if len(sel_areas)>0 else 1.0
                size_sim = np.exp(- ((np.log1p(areas) - np.log1p(mean_area))**2) / (2 * (0.6**2)))
            except Exception:
                size_sim = np.ones(X.shape[0])

        final_score = (
            WEIGHT_MAHAL * sim_from_mahal +
            WEIGHT_COS * ((cos_sim + 1.0) / 2.0) +
            WEIGHT_COLOR * color_sim +
            WEIGHT_SIZE * size_sim
        )

        cand_df['similarity'] = final_score
        # exclude already selected candidate rows (so we suggest only new ones)
        # Determine which rows correspond to selected examples (sel_idx)
        cand_df['is_example_selected'] = 0
        for i in sel_idx:
            if i < len(cand_df):
                cand_df.at[i, 'is_example_selected'] = 1

        suggestions = cand_df[(cand_df['is_example_selected'] == 0) & (cand_df['similarity'] >= SIMILARITY_THRESHOLD)]
        suggestions = suggestions.sort_values('similarity', ascending=False).head(MAX_RESULTS)

        out_df = suggestions[['label','centroid_x','centroid_y','similarity','orig_region_index']].copy()
        out_df.rename(columns={'centroid_x':'x','centroid_y':'y','similarity':'score','orig_region_index':'orig_index'}, inplace=True)
        csv_buf = out_df.to_csv(index=False)

        avg_sel_sim = cand_df.loc[sel_idx, 'similarity'].mean() if len(sel_idx)>0 else 0.0
        std_sel_sim = cand_df.loc[sel_idx, 'similarity'].std() if len(sel_idx)>0 else 0.0
        if len(suggestions) > 0:
            low_conf = (suggestions['similarity'] < max(0.0, avg_sel_sim - std_sel_sim)).sum()
            estimated_error_rate = float(low_conf) / float(len(suggestions))
        else:
            estimated_error_rate = 0.0

        response_payload = {
            "csv": csv_buf,
            "num_candidates": int(len(cand_df)),
            "num_selected_examples": int(len(sel_idx)),
            "num_suggestions": int(len(suggestions)),
            "estimated_error_rate": estimated_error_rate,
            "debug": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "selected_similarity_mean": float(avg_sel_sim),
                "selected_similarity_std": float(std_sel_sim),
                "weights": {
                    "mahal": WEIGHT_MAHAL,
                    "cos": WEIGHT_COS,
                    "color": WEIGHT_COLOR,
                    "size": WEIGHT_SIZE
                }
            },
            "suggestions_list": out_df.to_dict(orient="records")
        }

        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="auto_select_similar_cells",
            status="success",
            metadata={
                "candidates_count": len(cand_df),
                "examples_count": len(sel_idx),
                "suggestions_count": len(suggestions),
                "estimated_error_rate": estimated_error_rate
            }
        )

        return jsonify(response_payload)
    except Exception as e:
        ActivityLogger.log_activity(
            user_id=user_id,
            user_role=user_role,
            action_type="api_call",
            action_details="auto_select_similar_cells",
            status="error",
            metadata={"error": str(e)}
        )
        traceback.print_exc()
        return jsonify({"error":"internal_server_error", "message": str(e)}), 500
