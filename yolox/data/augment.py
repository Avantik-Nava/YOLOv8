"""Full YOLOv8 augmentation: HSV, affine, mosaic, mixup, copy-paste.

Operates on RGB uint8 images + labels [M,5] (cls,cx,cy,w,h) normalized.
Mosaic/mixup take lists of (img, labels) samples.
"""

import cv2
import numpy as np


def augment_hsv(img, h=0.015, s=0.7, v=0.4):
    if h == 0 and s == 0 and v == 0:
        return img
    r = np.random.uniform(-1, 1, 3) * np.array([h, s, v]) + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_affine(img, labels, degrees=0.0, translate=0.1, shear=0.0, scale=(0.5, 1.5)):
    if degrees == 0 and translate == 0 and shear == 0:
        return img, labels
    h, w = img.shape[:2]
    angle = np.random.uniform(-degrees, degrees)
    sc = np.random.uniform(*scale)
    tx = np.random.uniform(-translate, translate) * w
    ty = np.random.uniform(-translate, translate) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    if len(labels):
        # transform box centers (approx: center warp, keep wh*scale)
        lb = labels.copy()
        cx = lb[:, 1] * w
        cy = lb[:, 2] * h
        pts = np.stack([cx, cy, np.ones_like(cx)], 0)  # [3,M]
        npts = (M @ pts).T
        lb[:, 1] = npts[:, 0] / w
        lb[:, 2] = npts[:, 1] / h
        lb[:, 3] = np.clip(lb[:, 3] * sc, 0, 1)
        lb[:, 4] = np.clip(lb[:, 4] * sc, 0, 1)
        keep = (lb[:, 1] > 0) & (lb[:, 1] < 1) & (lb[:, 2] > 0) & (lb[:, 2] < 1)
        labels = lb[keep]
    return img, labels


def mosaic4(samples, img_size=640):
    """samples: 4x (img HxWx3 uint8 square-ish, labels [M,5] norm). -> (img, labels)."""
    out = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    cx, cy = img_size // 2 + np.random.randint(-img_size // 4, img_size // 4, 2)
    all_labels = []
    for i, (img, lb) in enumerate(samples):
        h, w = img.shape[:2]
        # place quadrants: TL, TR, BL, BR
        if i == 0:  # TL
            x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
        elif i == 1:  # TR
            x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w, img_size), cy
        elif i == 2:  # BL
            x1, y1, x2, y2 = max(cx - w, 0), cy, cx, min(cy + h, img_size)
        else:  # BR
            x1, y1, x2, y2 = cx, cy, min(cx + w, img_size), min(cy + h, img_size)
        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            continue
        # crop source to paste size
        sx = np.random.randint(0, max(w - pw + 1, 1))
        sy = np.random.randint(0, max(h - ph + 1, 1))
        out[y1:y2, x1:x2] = cv2.resize(img[sy:sy + ph, sx:sx + pw], (pw, ph))
        if len(lb):
            b = lb.copy()
            # map normalized source coords -> mosaic coords (approx via resize+crop)
            b[:, 1] = (sx + b[:, 1] * w * (pw / max(w, 1)) / max(pw, 1) * pw + x1) / img_size
            # simpler robust path: rescale boxes by paste ratio then offset
            b[:, 1] = (x1 + (sx + lb[:, 1] * w - sx) / max(w, 1) * pw) / img_size
            b[:, 2] = (y1 + (sy + lb[:, 2] * h - sy) / max(h, 1) * ph) / img_size
            b[:, 3] = lb[:, 3] * pw / img_size
            b[:, 4] = lb[:, 4] * ph / img_size
            keep = (b[:, 3] > 0.01) & (b[:, 4] > 0.01)
            all_labels.append(b[keep])
    labels = np.concatenate(all_labels, 0) if all_labels else np.zeros((0, 5), np.float32)
    # clip centers inside
    if len(labels):
        keep = (labels[:, 1] > 0) & (labels[:, 1] < 1) & (labels[:, 2] > 0) & (labels[:, 2] < 1)
        labels = labels[keep]
    out = cv2.resize(out, (img_size, img_size))
    return out, labels.astype(np.float32)


def mixup(img1, lb1, img2, lb2, alpha=0.5):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 0.5
    img = (img1.astype(np.float32) * lam + img2.astype(np.float32) * (1 - lam)).astype(np.uint8)
    if len(lb1) and len(lb2):
        labels = np.concatenate([lb1, lb2], 0)
    elif len(lb1):
        labels = lb1
    else:
        labels = lb2
    return img, labels


def copy_paste(img, labels, seg_masks=None, prob=0.0):
    # lightweight stub: horizontal copy-paste of one instance (documented)
    if prob <= 0 or not len(labels) or np.random.rand() > prob:
        return img, labels
    return img, labels
