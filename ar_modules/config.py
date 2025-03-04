# ar_modules/config.py
import cv2
import numpy as np

def get_orb_detector():
    # Load template for ORB detection
    template_path = 'gambar/kertas.jpg'
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        raise FileNotFoundError(f"Gambar referensi tidak ditemukan: {template_path}")
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    kp_template, des_template = orb.detectAndCompute(template_img, None)
    if des_template is None:
        raise ValueError("Deskripsi ORB untuk gambar referensi tidak ditemukan.")
    
    return orb, kp_template, des_template

def get_ar_image():
    # Load AR image for overlay
    ar_image = cv2.imread('gambar/1.png', cv2.IMREAD_UNCHANGED)
    if ar_image is None:
        raise FileNotFoundError("AR image not found: gambar/1.png")
    return ar_image

def get_lk_params():
    # Lucas-Kanade optical flow parameters
    return dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )