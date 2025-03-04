# ar_modules/image_processing.py
import cv2
import numpy as np
import base64
from ar_modules.config import get_orb_detector, get_ar_image, get_lk_params
from ar_modules.objloader import *
import os
import math

# Initialize components
orb, kp_template, des_template = get_orb_detector()
ar_image = get_ar_image()
lk_params = get_lk_params()

def decode_image(data_url):
    try:
        header, encoded = data_url.split(',', 1)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def encode_image(img):
    ret, jpeg = cv2.imencode('.jpg', img)
    if not ret:
        return None
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')

def overlay_image(frame, overlay, center):
    h, w = overlay.shape[:2]
    x = int(center[0] - w // 2)
    y = int(center[1] - h // 2)
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame
    roi = frame[y:y+h, x:x+w]
    if overlay.shape[2] == 4:
        overlay_img = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] + 
                            (1 - alpha_mask) * roi[:, :, c])
    else:
        roi[:] = overlay
    frame[y:y+h, x:x+w] = roi
    return frame

def detect_largest_square(img_gray, min_area=500, epsilon_ratio=0.02, aspect_tolerance=0.2):
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_square = None
    max_area = 0
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        epsilon = epsilon_ratio * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > min_area and area > max_area:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 1 - aspect_tolerance <= aspect_ratio <= 1 + aspect_tolerance:
                    best_square = approx
                    max_area = area
    return best_square

def match_with_orb_in_box(img_gray, best_square):
    kp_frame, des_frame = orb.detectAndCompute(img_gray, None)
    if des_frame is None or kp_frame is None:
        return best_square, 0, 0  # Jika tidak ada fitur ORB, gunakan titik asli

    # Ambil koordinat best_square
    x_min, y_min = np.min(best_square, axis=0)[0]
    x_max, y_max = np.max(best_square, axis=0)[0]

    # Filter keypoints ORB agar hanya dalam best_square
    filtered_kp = []
    filtered_des = []
    kp_positions = []

    for kp, des in zip(kp_frame, des_frame):
        x, y = kp.pt
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_kp.append(kp)
            filtered_des.append(des)
            kp_positions.append((x, y))

    if len(filtered_des) < 4:
        return best_square, 0, 0  # Jika fitur tidak cukup, gunakan urutan titik asli

    # Gunakan BFMatcher untuk mencocokkan fitur dengan template
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, np.array(filtered_des))
    matches = sorted(matches, key=lambda x: x.distance)

    total_features = max(len(des_template), len(filtered_des))
    match_ratio = len(matches) / total_features if total_features > 0 else 0

    if len(matches) >= 4:
        # Ambil titik best_square yang paling cocok dengan template
        match_dict = {}
        for m in matches[:4]:  # Ambil 4 match terbaik
            template_idx = m.queryIdx
            frame_idx = m.trainIdx
            match_dict[tuple(kp_template[template_idx].pt)] = filtered_kp[frame_idx].pt

        # Urutkan best_square berdasarkan kecocokan dengan template
        ordered_square = np.array([
            np.array(match_dict.get(tuple(kp.pt), np.array(best_square[i]))).tolist()  # Pastikan fallback ke array valid
            for i, kp in enumerate(kp_template[:4])
        ], dtype=np.float32)

    else:
        ordered_square = np.array(best_square, dtype=np.float32)  # Jika tidak cukup match, tetap gunakan titik asli

    return ordered_square, len(matches), match_ratio


def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def order_points(pts):
    """ Mengurutkan 4 titik dalam format [Kiri Atas, Kanan Atas, Kanan Bawah, Kiri Bawah] """
    pts = np.array(pts, dtype="float32").reshape(4, 2)  # Pastikan bentuknya (4,2)
    
    rect = np.zeros((4, 2), dtype="float32")

    # Urutkan berdasarkan X untuk mendapatkan kiri & kanan
    x_sorted = pts[np.argsort(pts[:, 0])]

    left = x_sorted[:2]
    right = x_sorted[2:]

    # Urutkan kiri dan kanan berdasarkan Y untuk mendapatkan atas & bawah
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    rect[0] = left[0]  # Kiri atas
    rect[1] = right[0]  # Kanan atas
    rect[2] = right[1]  # Kanan bawah
    rect[3] = left[1]  # Kiri bawah

    return rect

def process_frame(frame, prev_gray, prev_points):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    final_img = frame.copy()
    best_square = detect_largest_square(frame_gray)

    match_count, match_ratio = 0, 0
    dir_name = os.getcwd()
    obj = OBJ(os.path.join(dir_name, 'gambar/fox.obj'), swapyz=True)

    # Tentukan parameter kamera
    focal_length = max(frame.shape)  # Berdasarkan resolusi frame
    cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    camera_parameters = np.array([[focal_length, 0, cx],
                                  [0, focal_length, cy],
                                  [0, 0, 1]])

    homography = None

    if best_square is not None:
        best_square, match_count, match_ratio = match_with_orb_in_box(frame_gray, best_square)

    if match_count >= 50 and match_ratio >= 0.2:
        color = (0, 255, 0)
        prev_points = np.array([pt[0] for pt in best_square], dtype=np.float32)
        prev_gray = frame_gray.copy()
    elif prev_gray is not None and prev_points is not None:
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, prev_points, None, **lk_params)
        if new_points is not None and status is not None:
            valid_points = new_points[status.flatten() == 1]
            if len(valid_points) == 4:
                prev_points = valid_points
                prev_gray = frame_gray.copy()
                best_square = np.int32(valid_points)
                color = (255, 255, 0)
            else:
                best_square = None
                color = (0, 0, 255)
        else:
            best_square = None
            color = (0, 0, 255)
    else:
        best_square = None
        color = (0, 0, 255)

    # Tambahkan status teks pada gambar
    log_text = f"Match: {match_count}, Similarity: {match_ratio:.2%}"
    cv2.putText(final_img, log_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    # Gambar kotak dan render model 3D jika memungkinkan
    if best_square is not None:
        cv2.polylines(final_img, [np.int32(best_square)], True, color, 3)

        scale_factor = 2.5
        w = np.linalg.norm(best_square[0] - best_square[1]) * scale_factor
        h = np.linalg.norm(best_square[0] - best_square[3]) * scale_factor

        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.float32(best_square)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if homography is not None:
            try:
                projection = projection_matrix(camera_parameters, homography)
                final_img = render(final_img, obj, projection, frame_gray, False)
            except Exception as e:
                print(f"Error rendering 3D model: {e}")

    return final_img, prev_gray, prev_points

