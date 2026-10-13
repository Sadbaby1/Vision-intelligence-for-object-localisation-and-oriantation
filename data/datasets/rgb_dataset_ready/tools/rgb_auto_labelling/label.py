import cv2, numpy as np, os, json
from detect import fs, comps
from frames import find_frames

def white_mask(hsv):
    m = ((hsv[..., 1] < 45) & (hsv[..., 2] > 150)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

def find_filters(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = white_mask(hsv)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 1500: continue
        c = max(cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        r = cv2.minAreaRect(c); (cx, cy), (w, h), a = r
        L, W = max(w, h), min(w, h); sol = st[i, cv2.CC_STAT_AREA] / (w * h + 1e-6)
        if cx < 330: continue                      # static metal rail / bench on the left
        if 110 <= L <= 300 and 18 <= W <= 75 and sol > 0.55:
            x, y, bw, bh = cv2.boundingRect(c)
            out.append(dict(rect=r, L=L, W=W, sol=sol, handheld=bool(cy > 500), edge=bool(y + bh >= 715)))
    return out

# Manual correction: in this image the frame's left end hangs over the cardboard edge,
# so colour segmentation splits it. Corners read by eye from a zoomed, gridded crop.
OVERRIDES = {16: np.array([[305, 403], [562, 372], [567, 434], [307, 463]], np.float32)}

def frames_of(im, k=None):
    o, _ = find_frames(im)
    if k in OVERRIDES:
        r = cv2.minAreaRect(OVERRIDES[k]); (cx, cy), (w, h), a = r
        keep = [c for c in o if c['cy'] < 300 and 220 <= c['L'] <= 290 and 50 <= c['W'] <= 85]  # table frame only
        return [dict(rect=c['rect'], L=c['L'], W=c['W'], sol=c['fill'], handheld=False, edge=False) for c in keep] + \
               [dict(rect=r, L=max(w, h), W=min(w, h), sol=float('nan'), handheld=False, edge=False, manual=True)]
    return [dict(rect=c['rect'], L=c['L'], W=c['W'], sol=c['fill'], handheld=False, edge=False)
            for c in o if 220 <= c['L'] <= 290 and 50 <= c['W'] <= 85 and c['fill'] < 0.45]

def to_line(cls, rect, w=1280, h=720):
    pts = cv2.boxPoints(rect)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1) / w
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1) / h
    return f"{cls} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))

if __name__ == '__main__':
    os.makedirs('labels', exist_ok=True); os.makedirs('check', exist_ok=True)
    rows = []
    for k, f in enumerate(fs):
        im = cv2.imread(f); name = os.path.splitext(os.path.basename(f))[0]
        dets = [(0, d) for d in find_filters(im)] + [(1, d) for d in frames_of(im, k)]
        with open(f'labels/{name}.txt', 'w') as fh:
            fh.write("\n".join(to_line(c, d['rect']) for c, d in dets) + "\n")
        vis = im.copy()
        for c, d in dets:
            col = (255, 80, 0) if c == 0 else (0, 0, 255)
            cv2.drawContours(vis, [np.int32(cv2.boxPoints(d['rect']))], -1, col, 2)
            rows.append(dict(image=name, cls=['filter', 'frame'][c], L_px=round(d['L'], 1), W_px=round(d['W'], 1),
                             fill=round(d['sol'], 2), handheld=d['handheld'], cut_at_edge=d['edge'], manual=d.get('manual', False)))
        cv2.putText(vis, str(k), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        cv2.imwrite(f'check/{name}.png', vis)
        print(k, [(['F', 'R'][c], int(d['L']), int(d['W'])) for c, d in dets])
    json.dump(rows, open('dets.json', 'w'), indent=1)
