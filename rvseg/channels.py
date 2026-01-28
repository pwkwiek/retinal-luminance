import numpy as np
import cv2

def ch_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

def ch_rgb_R(bgr): return bgr[...,2].astype(np.float32) / 255.0
def ch_rgb_G(bgr): return bgr[...,1].astype(np.float32) / 255.0
def ch_rgb_B(bgr): return bgr[...,0].astype(np.float32) / 255.0

def ch_yuv_Y(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    return yuv[...,0].astype(np.float32) / 255.0
def ch_yuv_U(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    return yuv[...,1].astype(np.float32) / 255.0
def ch_yuv_V(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    return yuv[...,2].astype(np.float32) / 255.0

def ch_ycrcb_Y(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycc[...,0].astype(np.float32) / 255.0
def ch_ycrcb_Cr(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycc[...,1].astype(np.float32) / 255.0
def ch_ycrcb_Cb(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycc[...,2].astype(np.float32) / 255.0

def ch_hsv_H(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[...,0].astype(np.float32) / 179.0
def ch_hsv_S(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[...,1].astype(np.float32) / 255.0
def ch_hsv_V(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[...,2].astype(np.float32) / 255.0

def ch_hsl_H(bgr):
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    return hls[...,0].astype(np.float32) / 179.0
def ch_hsl_L(bgr):
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    return hls[...,1].astype(np.float32) / 255.0
def ch_hsl_S(bgr):
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    return hls[...,2].astype(np.float32) / 255.0

def ch_lab_L(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[...,0].astype(np.float32) / 255.0
def ch_lab_a(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[...,1].astype(np.float32) / 255.0
def ch_lab_b(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[...,2].astype(np.float32) / 255.0

CHANNELS = {
    "GRAY": ch_gray,
    "RGB_R": ch_rgb_R,
    "RGB_G": ch_rgb_G,
    "RGB_B": ch_rgb_B,
    "YUV_Y": ch_yuv_Y,
    "YUV_U": ch_yuv_U,
    "YUV_V": ch_yuv_V,
    "HSV_H": ch_hsv_H,
    "HSV_S": ch_hsv_S,
    "HSV_V": ch_hsv_V,
    "HSL_H": ch_hsl_H,
    "HSL_L": ch_hsl_L,
    "HSL_S": ch_hsl_S,
    "YCrCb_Y": ch_ycrcb_Y,
    "YCrCb_Cr": ch_ycrcb_Cr,
    "YCrCb_Cb": ch_ycrcb_Cb,
    "Lab_L": ch_lab_L,
    "Lab_a": ch_lab_a,
    "Lab_b": ch_lab_b,
}

def ch_weighted_rgb(bgr, wR, wG, wB):
    B = bgr[...,0].astype(np.float32) / 255.0
    G = bgr[...,1].astype(np.float32) / 255.0
    R = bgr[...,2].astype(np.float32) / 255.0
    return np.clip(wR*R + wG*G + wB*B, 0.0, 1.0)