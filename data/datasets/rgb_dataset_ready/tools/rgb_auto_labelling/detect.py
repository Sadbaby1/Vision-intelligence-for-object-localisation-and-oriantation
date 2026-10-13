import cv2, numpy as np, glob, json
fs = sorted(glob.glob('to_label/*.png'))

def comps(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area: continue
        m = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), (w, h), a = cv2.minAreaRect(c)
        L, W = max(w, h), min(w, h)
        out.append(dict(cx=cx, cy=cy, L=L, W=W, ang=a, area=st[i, cv2.CC_STAT_AREA],
                        fill=cv2.contourArea(c) / (w * h + 1e-6), sol=st[i, cv2.CC_STAT_AREA]/(w*h+1e-6)))
    return out

if __name__ == '__main__':
    for k, f in enumerate(fs):
        im = cv2.imread(f); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 150)).astype(np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cs = [c for c in comps(white, 1500)]
        print(k, [(int(c['cx']), int(c['cy']), int(c['L']), int(c['W']), round(c['sol'], 2)) for c in cs])
