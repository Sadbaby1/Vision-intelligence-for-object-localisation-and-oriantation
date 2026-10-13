import cv2, numpy as np, glob
from detect import fs
def light_roi(hsv):
    h,s,v = hsv[...,0],hsv[...,1],hsv[...,2]
    light = ((h>=8)&(h<=30)&(s>40)&(v>140)).astype(np.uint8)
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
    n,lab,st,_ = cv2.connectedComponentsWithStats(light)
    i = 1+np.argmax(st[1:,cv2.CC_STAT_AREA])
    m = (lab==i).astype(np.uint8)
    cnts,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(m); cv2.drawContours(roi,[max(cnts,key=cv2.contourArea)],-1,1,-1)  # fill holes
    return roi
def find_frames(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    roi = light_roi(hsv)
    roi = cv2.erode(roi, np.ones((5,5),np.uint8))
    v,sat=hsv[...,2],hsv[...,1]
    dark = (((v < 140) | ((sat < 40) & (v < 165))) & (roi>0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n,lab,st,_ = cv2.connectedComponentsWithStats(dark)
    out=[]
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA] < 800: continue
        m=(lab==i).astype(np.uint8)
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        c=max(cnts,key=cv2.contourArea)
        r=cv2.minAreaRect(c); (cx,cy),(w,h),a=r
        L,W=max(w,h),min(w,h)
        out.append(dict(rect=r,cnt=c,cx=cx,cy=cy,L=L,W=W,fill=st[i,cv2.CC_STAT_AREA]/(w*h+1e-6)))
    return out, roi
if __name__=='__main__':
    for k,f in enumerate(fs):
        o,_=find_frames(cv2.imread(f))
        print(k,[(int(c['cx']),int(c['cy']),int(c['L']),int(c['W']),round(c['fill'],2)) for c in o])
