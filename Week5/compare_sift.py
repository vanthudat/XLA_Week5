import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

# A repeatable, textured scene and a rotated/rescaled second view.
def make_images():
    rng = np.random.default_rng(7)
    im = np.full((420, 600), 35, dtype=np.uint8)
    for _ in range(85):
        x, y = rng.integers(25, 575), rng.integers(25, 395)
        radius = int(rng.integers(3, 13))
        col = int(rng.integers(85, 245))
        cv2.circle(im, (int(x), int(y)), radius, col, -1, cv2.LINE_AA)
    cv2.rectangle(im, (100, 90), (270, 240), 230, 3)
    cv2.line(im, (95, 245), (280, 85), 210, 3)
    cv2.putText(im, 'SIFT', (300, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3, cv2.LINE_AA)
    cv2.putText(im, '2026', (320, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 190, 3, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((300, 210), 17, 0.90)
    M[:, 2] += [8, -5]
    transformed = cv2.warpAffine(im, M, (600, 420), flags=cv2.INTER_LINEAR, borderValue=35)
    return im, transformed

# Educational implementation of the main SIFT pipeline (not an exact OpenCV clone).
class SimpleSIFT:
    def __init__(self, octaves=3, scales=3, sigma=1.6, contrast=3.5, edge_r=10, max_points=700):
        self.octaves, self.scales, self.sigma = octaves, scales, sigma
        self.contrast, self.edge_r, self.max_points = contrast, edge_r, max_points

    def pyramid(self, gray):
        base = gray.astype(np.float32)
        pyramids = []
        for o in range(self.octaves):
            # Gaussian images have sigma = sigma*2^(s/S) within each octave.
            gs = [cv2.GaussianBlur(base, (0, 0), self.sigma * (2 ** (s / self.scales)))
                  for s in range(self.scales + 3)]
            dogs = [gs[s + 1] - gs[s] for s in range(len(gs) - 1)]
            pyramids.append((gs, dogs))
            base = cv2.resize(base, (base.shape[1] // 2, base.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return pyramids

    def find_points(self, pyramids):
        candidates = []
        for octave, (_, dogs) in enumerate(pyramids):
            for s in range(1, self.scales + 1):
                prev, cur, nex = dogs[s - 1:s + 2]
                h, w = cur.shape
                for y in range(9, h - 9):
                    for x in range(9, w - 9):
                        v = cur[y, x]
                        if abs(v) < self.contrast:
                            continue
                        cube = np.stack([a[y-1:y+2, x-1:x+2] for a in (prev, cur, nex)])
                        vals = np.delete(cube.reshape(-1), 13)
                        if not (v > vals.max() or v < vals.min()):
                            continue
                        # Hessian edge-response rejection in the current DoG level.
                        dxx = cur[y, x+1] + cur[y, x-1] - 2*v
                        dyy = cur[y+1, x] + cur[y-1, x] - 2*v
                        dxy = (cur[y+1, x+1] - cur[y+1, x-1] - cur[y-1, x+1] + cur[y-1, x-1]) / 4
                        det = dxx*dyy - dxy*dxy
                        if det <= 0 or (dxx+dyy)**2 / det >= (self.edge_r+1)**2/self.edge_r:
                            continue
                        candidates.append((abs(float(v)), octave, s, x, y))
        candidates.sort(reverse=True)
        return candidates[:self.max_points]

    @staticmethod
    def orientation(blur, x, y, radius=8):
        patch = blur[y-radius:y+radius+1, x-radius:x+radius+1]
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
        weights = np.exp(-(xx*xx+yy*yy)/(2*(0.65*radius)**2))
        hist = np.zeros(36, np.float32)
        np.add.at(hist, (ang / 10).astype(int).ravel() % 36, (mag*weights).ravel())
        return float(hist.argmax()*10)

    @staticmethod
    def descriptor(blur, x, y, angle, step=4):
        # 4x4 cells, 8 orientation bins each; oriented 16x16 sampling region.
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        hist = np.zeros((4, 4, 8), dtype=np.float32)
        h, w = blur.shape
        for row in range(-8, 8):
            for col in range(-8, 8):
                rx = c*col-s*row + x
                ry = s*col+c*row + y
                ix, iy = int(round(rx)), int(round(ry))
                if not (1 <= ix < w-1 and 1 <= iy < h-1):
                    return None
                gx = float(blur[iy, ix+1] - blur[iy, ix-1])
                gy = float(blur[iy+1, ix] - blur[iy-1, ix])
                mag = np.hypot(gx, gy)
                a = (np.degrees(np.arctan2(gy, gx)) - angle) % 360
                bin_idx = min(int(a / 45), 7)
                cy, cx = (row+8)//step, (col+8)//step
                weight = np.exp(-(row*row+col*col)/(2*8.0**2))
                hist[cy, cx, bin_idx] += mag*weight
        desc = hist.reshape(-1)
        desc /= np.linalg.norm(desc) + 1e-7
        desc = np.clip(desc, 0, 0.2)
        desc /= np.linalg.norm(desc) + 1e-7
        return desc

    def detectAndCompute(self, gray, mask=None):
        pyr = self.pyramid(gray)
        candidates = self.find_points(pyr)
        keypoints, descriptors = [], []
        for _, o, level, x, y in candidates:
            blur = pyr[o][0][level]
            # Allow the 16x16 sampling region to rotate without going out of bounds.
            if x < 13 or y < 13 or x >= blur.shape[1]-13 or y >= blur.shape[0]-13:
                continue
            a = self.orientation(blur, x, y)
            d = self.descriptor(blur, x, y, a)
            if d is None:
                continue
            mult = 2**o
            kp = cv2.KeyPoint(float(x*mult), float(y*mult), float(self.sigma*2**(level/self.scales)*mult*2), a)
            keypoints.append(kp)
            descriptors.append(d)
        descs = np.asarray(descriptors, dtype=np.float32).reshape(-1, 128)
        return keypoints, descs


def matches(d1, d2, ratio=0.75):
    if len(d1) == 0 or len(d2) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    return sorted([m for m, n in pairs if m.distance < ratio*n.distance], key=lambda m: m.distance)


def evaluate(name, detector, im1, im2):
    start = time.perf_counter()
    k1, d1 = detector.detectAndCompute(im1, None)
    k2, d2 = detector.detectAndCompute(im2, None)
    elapsed = (time.perf_counter()-start)*1000
    good = matches(d1, d2)
    print(f'{name:17} | image1: {len(k1):4d} {str(d1.shape):12} | image2: {len(k2):4d} {str(d2.shape):12} | matches: {len(good):4d} | time: {elapsed:.1f} ms')
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    vis1 = cv2.drawKeypoints(im1, k1, None, flags=flags)
    vis2 = cv2.drawMatches(im1, k1, im2, k2, good[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(OUT / f'{name}_keypoints.jpg'), vis1)
    cv2.imwrite(str(OUT / f'{name}_matches.jpg'), vis2)
    return len(k1), len(k2), len(good), elapsed

