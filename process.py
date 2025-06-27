import sys
import os
from functools import partial

import cv2
import numpy as np

class ImageProcessor:
    #大作业第一部分
    @staticmethod
    def invert(img: np.ndarray):
        return 255 - img

    @staticmethod
    def add(img1: np.ndarray, img2: np.ndarray,
            alpha: float = 0.5, resize_mode: str = "min"):

        if img1.ndim == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if resize_mode == "scale" and (h1 != h2 or w1 != w2):
            img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
        else: 
            h, w = min(h1, h2), min(w1, w2)
            img1 = img1[:h, :w]
            img2 = img2[:h, :w]

        alpha = np.clip(alpha, 0.0, 1.0)
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        out = img1_f * alpha + img2_f * (1.0 - alpha)

        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def translate(img: np.ndarray, dx: int, dy: int):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    @staticmethod
    def rotate(img: np.ndarray, angle: float):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def mirror(img: np.ndarray, axis):
        return cv2.flip(img, axis)
    
    #大作业第二部分
    @staticmethod
    def fft2(img: np.ndarray):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f_rows = np.fft.fft(gray.astype(np.float32), axis=1) 
        f = np.fft.fft(f_rows, axis=0)
        
        # Step1
        mag1 = np.abs(f)
        disp1 = cv2.cvtColor(
            cv2.normalize(mag1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )

        # Step2
        fshift = np.fft.fftshift(f)
        mag2 = np.abs(fshift)
        disp2 = cv2.cvtColor(
            cv2.normalize(mag2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )

        # Step3
        mag3 = 2*np.log(mag2 + 1)
        disp3 = cv2.cvtColor(
            cv2.normalize(mag3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )

        return disp1, disp2, disp3, fshift

    @staticmethod
    def ifft2(_, fshift=None):
        if fshift is None:
            return _
        ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)

        img_back = cv2.normalize(img_back, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def conjugate_dft_inverse(img):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = gray.astype(np.float32)
        rows, cols = gray.shape

        centering = np.fromfunction(lambda x, y: (-1)**(x + y), (rows, cols))
        centered = gray * centering
        f = np.fft.fft2(centered)
        f_conj = np.conj(f)
        img_back = np.fft.ifft2(f_conj)
        result = np.real(img_back) * centering
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    #大作业第三部分
    @staticmethod
    def hist_eq(img: np.ndarray):

        if img.ndim == 2:
            eq1 = cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            eq1 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return eq1

    @staticmethod
    def homomorphic(img: np.ndarray, gamma_l=0.5, gamma_h=1.8, c=1.0, d0=30):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = gray.astype(np.float32) / 255.0
        rows, cols = gray.shape
        gray_log = np.log(gray + 1e-6)
        F = np.fft.fftshift(np.fft.fft2(gray_log))

        u = np.arange(-rows//2, rows//2)
        v = np.arange(-cols//2, cols//2)
        V, U = np.meshgrid(v, u)
        D2 = U**2 + V**2
        H = (gamma_h - gamma_l) * (1 - np.exp(-c * D2 / (d0**2))) + gamma_l

        G = H * F
        g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
        g_exp = np.exp(g)
        result = cv2.normalize(g_exp, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)

    @staticmethod
    def exp_transform(img: np.ndarray, gamma=1.2):
        def _exp(x):
            x_f = x.astype(np.float32) / 255.0
            x_e = np.power(x_f, gamma)
            return (x_e * 255).clip(0, 255).astype(np.uint8)

        if img.ndim == 2:
            return _exp(img)
        else:
            b, g, r = cv2.split(img)
            return cv2.merge([_exp(b), _exp(g), _exp(r)])

    @staticmethod
    def laplace_sharpen(img: np.ndarray):
        lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
        sharp = cv2.convertScaleAbs(lap)
        return cv2.addWeighted(img, 1.0, sharp, 1.0, 0)
    
    #大作业第四部分
    @staticmethod
    def edge_roberts(img: np.ndarray):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = g.astype(np.float32)
        kx = np.array([[1, 0],
                       [0,-1]], dtype=np.float32)
        ky = np.array([[0, 1],
                       [-1,0]], dtype=np.float32)
        gx = cv2.filter2D(g, -1, kx)
        gy = cv2.filter2D(g, -1, ky)
        mag = np.sqrt(gx**2 + gy**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_prewitt(img: np.ndarray):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = g.astype(np.float32)
        kx = np.array([[ 1, 0,-1],
                       [ 1, 0,-1],
                       [ 1, 0,-1]], dtype=np.float32)
        ky = np.array([[ 1,  1,  1],
                       [ 0,  0,  0],
                       [-1, -1, -1]], dtype=np.float32)
        gx = cv2.filter2D(g, -1, kx)
        gy = cv2.filter2D(g, -1, ky)
        mag = np.sqrt(gx**2 + gy**2)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_sobel(img: np.ndarray):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def edge_laplace(img: np.ndarray):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        mag = np.abs(lap)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    

    #大作业第五部分
    @staticmethod
    def largest_contour(gray: np.ndarray, min_area=50):
        H, W = gray.shape
        img_area = H * W

        bin_imgs = []
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bin_imgs.extend([th1, th2])

        cand = []
        for bin_img in bin_imgs:
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                if area >= 0.95 * img_area:
                    continue
                cand.append((area, c))

        if not cand:
            return None

        pts = max(cand, key=lambda x: x[0])[1][:, 0, :]
        return pts.astype(np.float32)

    @staticmethod
    def resample_contour(pts: np.ndarray, N: int = 64):
        pts = pts.astype(np.float32)

        diff = np.diff(np.vstack([pts, pts[0]]), axis=0)
        seg_len = np.linalg.norm(diff, axis=1)
        cum_len = np.insert(np.cumsum(seg_len), 0, 0)
        total = cum_len[-1]
        new_s = np.linspace(0, total, N, endpoint=False)
        new_pts = np.empty((N, 2), np.float32)
        for i, s in enumerate(new_s):
            idx = np.searchsorted(cum_len, s, side="right") - 1
            t = (s - cum_len[idx]) / seg_len[idx]
            new_pts[i] = pts[idx] + t * diff[idx]
        return new_pts

    @staticmethod
    def fourier_reconstruct(img: np.ndarray,
                            N: int = 64,
                            Ms=(62, 32, 2)):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        cont = ImageProcessor.largest_contour(gray)
        if cont is None:
            return []

        cont = ImageProcessor.resample_contour(cont, N) 
        z = cont[:, 0] + 1j * cont[:, 1]      
        C = np.fft.fft(z) / N       

        def _rebuild(M):
            keep = np.zeros_like(C)
            half = M // 2
            keep[:half+1] = C[:half+1]
            keep[-half:] = C[-half:]
            z_rec = np.fft.ifft(keep * N)      
            return np.column_stack([z_rec.real, z_rec.imag]).astype(np.float32)

        results = [("原始轮廓 (N=100)", cont)]
        for m in Ms:
            results.append((f"重构 M={m}", _rebuild(m)))
        return results
    
    #大作业第六部分
    @staticmethod
    def dilate_once_inplace(img: np.ndarray, kernel_size=3):
        img_inv = cv2.bitwise_not(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(img_inv, kernel, iterations=1)
        img[:, :] = cv2.bitwise_not(dilated)

    @staticmethod
    def erode_once_inplace(img: np.ndarray, kernel_size=3):
        img_inv = cv2.bitwise_not(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded = cv2.erode(img_inv, kernel, iterations=1)
        img[:, :] = cv2.bitwise_not(eroded)

    @staticmethod
    def dilate_repeated(img: np.ndarray, kernel_size=3, iterations=10):
        img_inv = cv2.bitwise_not(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(img_inv, kernel, iterations=iterations)
        return cv2.bitwise_not(dilated)

    @staticmethod
    def erode_repeated(img: np.ndarray, kernel_size=3, iterations=10):
        img_inv = cv2.bitwise_not(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded = cv2.erode(img_inv, kernel, iterations=iterations)
        return cv2.bitwise_not(eroded)
    
    #大作业第七部分
    @staticmethod
    def sift_match(img1: np.ndarray,
                   img2: np.ndarray,
                   max_matches: int = 50,
                   ratio: float = 0.75):

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        good = sorted(good, key=lambda x: x.distance)[:max_matches]

        vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return vis