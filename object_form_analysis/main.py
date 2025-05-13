import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

try:
    from PIL.Image import Resampling
    RESAMPLE = Resampling.LANCZOS
except ImportError:
    RESAMPLE = Image.ANTIALIAS

def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def preprocess_mask(img, bin_thr, closing_it):
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5,5), 0)
    _, t = cv2.threshold(blur, bin_thr, 255, cv2.THRESH_BINARY_INV)
    return cv2.morphologyEx(t, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
                            iterations=closing_it)

def split_overlapping(mask, open_it, dilate_it, dist_frac):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_it)
    dist = cv2.distanceTransform(op, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, dist_frac * dist.max(), 255, cv2.THRESH_BINARY)
    fg = fg.astype(np.uint8)
    n, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[cv2.subtract(cv2.dilate(op, k, iterations=dilate_it), fg) == 255] = 0
    return markers, n + 1

def find_cards(img, mask, open_it, dilate_it, dist_frac):
    work = img.copy()
    markers, n = split_overlapping(mask, open_it, dilate_it, dist_frac)
    cv2.watershed(work, markers)
    cards = []
    for m in range(2, n):
        comp = (markers == m).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 1000:
            continue
        cards.append(c)
    return cards, markers

def curvature_peaks(contour, frac_thresh=0.3):
    pts = contour[:,0,:].astype(np.float32)
    v1 = pts - np.roll(pts,1,axis=0); nxt = np.roll(pts,-1,axis=0); v2 = nxt - pts
    v1n = v1 / (np.linalg.norm(v1,axis=1,keepdims=True)+1e-3)
    v2n = v2 / (np.linalg.norm(v2,axis=1,keepdims=True)+1e-3)
    angles = np.arccos(np.clip((v1n*v2n).sum(axis=1), -1,1))
    peaks = 0
    for i in range(1, len(angles)-1):
        if angles[i] > angles[i-1] and angles[i] > angles[i+1] and angles[i] > (frac_thresh * angles.max()):
            peaks += 1
    return peaks

def annotate_shapes(orig, cards, canny_lo, canny_hi, min_area, max_frac, peak_frac, peak_count_thresh):
    out = orig.copy()
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    for idx, cnt in enumerate(cards, start=1):
        cv2.drawContours(out, [cnt], -1, (0,255,0),2)
        M = cv2.moments(cnt)
        if M['m00']>0:
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        else:
            cx,cy = cnt[0][0]
        cv2.putText(out, f"#{idx}", (cx-15,cy+15), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
        x,y,w,h = cv2.boundingRect(cnt)
        closed = cv2.morphologyEx(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(orig[y:y+h, x:x+w],cv2.COLOR_BGR2GRAY),
                                                             (5,5),0), canny_lo, canny_hi), cv2.MORPH_CLOSE, k, iterations=1)
        cnts2,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good = []
        for c2 in cnts2:
            A = cv2.contourArea(c2)
            bx,by,bw,bh = cv2.boundingRect(c2)
            if A<min_area or A>max_frac*(w*h):
                continue
            if bx<=2 or by<=2 or bx+bw>=w-2 or by+bh>=h-2:
                continue
            good.append(c2)
        if not good:
            continue
        shape = max(good, key=cv2.contourArea)
        if curvature_peaks(shape, frac_thresh=peak_frac) <= peak_count_thresh:
            if len(shape)>=5:
                ellipse = cv2.fitEllipse(shape)
                center,axes,ang = ellipse
                center = (int(center[0])+x, int(center[1])+y)
                axes   = (int(axes[0]), int(axes[1]))
                cv2.ellipse(out, (center, axes, ang), (0,0,255),2)
            marker = "Smooth"
            M2 = cv2.moments(shape)
            if M2['m00']>0:
                px = int(M2['m10']/M2['m00'])+x
                py = int(M2['m01']/M2['m00'])+y
            else:
                px,py = x+w//2, y+h//2
        else:
            P = cv2.arcLength(shape,True)
            approx = cv2.approxPolyDP(shape, 0.02*P, True)
            approx_s = approx + np.array([[x,y]])
            cv2.drawContours(out, [approx_s], -1,(0,0,255),2)
            marker = f"P{len(approx)}" + ("C" if cv2.isContourConvex(approx) else "")
            M2 = cv2.moments(shape)
            if M2['m00']>0:
                px = int(M2['m10']/M2['m00'])+x
                py = int(M2['m01']/M2['m00'])+y
            else:
                px,py = x+w//2, y+h//2
        cv2.putText(out, marker, (px-10, py-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    return out

class CardsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа №2")
        self.default = {
            'bin_thr':100, 'closing':10,
            'open_it':2, 'dilate_it':10, 'dist_frac':50,
            'canny_lo':28, 'canny_hi':68,
            'min_area':300, 'max_frac':70,
            'peak_frac':0.65, 'peak_count_thresh':0
        }
        self.vars = {k: tk.DoubleVar(value=v) for k,v in self.default.items()}
        self.img_path = ''
        self.last_images = {}
        self._build_left()
        self._build_right()

    def _build_left(self):
        lf = tk.Frame(self); lf.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        tk.Button(lf, text="Выбрать файл…", command=self.select_image).pack(fill=tk.X, pady=2)
        tk.Button(lf, text="Сбросить", command=self.reset_defaults).pack(fill=tk.X, pady=2)
        params = [
            ('bin_thr',0,255,'int'),
            ('closing',0,10,'int'),
            ('open_it',0,10,'int'),
            ('dilate_it',0,10,'int'),
            ('dist_frac',0,100,'int'),
            ('canny_lo',0,255,'int'),
            ('canny_hi',0,255,'int'),
            ('min_area',0,10000,'int'),
            ('max_frac',0,100,'int'),
            ('peak_frac',0.0,1.0,'float'),
            ('peak_count_thresh',0,10,'int'),
        ]
        for name,mn,mx,typ in params:
            fr = tk.Frame(lf); fr.pack(fill=tk.X, pady=1)
            tk.Label(fr, text=name).pack(side=tk.LEFT)
            if typ == 'int':
                tk.Scale(fr, from_=mn, to=mx, orient=tk.HORIZONTAL, variable=self.vars[name])\
                  .pack(fill=tk.X, expand=True)
            else:
                tk.Scale(fr, from_=mn, to=mx, resolution=0.01, orient=tk.HORIZONTAL, variable=self.vars[name]).pack(fill=tk.X, expand=True)
        tk.Button(lf, text="Обработать", command=self.process).pack(fill=tk.X, pady=5)
        self.save_btn = tk.Button(lf, text="Сохранить", command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=2)

    def reset_defaults(self):
        for k,v in self.default.items():
            self.vars[k].set(v)

    def _build_right(self):
        rf = tk.Frame(self); rf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.labels = {}
        for i, title in enumerate(['Исходник','Маска','Ядра','Результат']):
            tk.Label(rf, text=title).grid(row=(i//2)*2, column=i%2, padx=5, pady=5)
            lbl = tk.Label(rf)
            lbl.grid(row=(i//2)*2+1, column=i%2, padx=5, pady=5)
            self.labels[title] = lbl

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image','*.jpg *.png *.bmp')])
        if not path: return
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror('Ошибка','Не удалось загрузить')
            return
        self.img_path = path
        self.orig = img
        self._show_img('Исходник', img)

    def process(self):
        if not hasattr(self,'orig'):
            messagebox.showwarning('Внимание','Сначала выберите файл')
            return
        v = {k: self.vars[k].get() for k in self.vars}
        mask = preprocess_mask(self.orig,
                               int(v['bin_thr']), int(v['closing']))
        cards, markers = find_cards(
            self.orig, mask,
            int(v['open_it']), int(v['dilate_it']), v['dist_frac']/100.0
        )
        res = annotate_shapes(self.orig, cards,
                              int(v['canny_lo']), int(v['canny_hi']),
                              int(v['min_area']), v['max_frac']/100.0,
                              v['peak_frac'], int(v['peak_count_thresh']))
        self._show_img('Маска', mask, is_gray=True)
        self._show_img('Ядра', markers.astype(np.uint8)*10, is_gray=True)
        self._show_img('Результат', res)
        self.last_images = {
            'Исходник': self.orig,
            'Маска': mask,
            'Ядра': markers.astype(np.uint8)*10,
            'Результат': res
        }
        self.save_btn.config(state=tk.NORMAL)

    def save_results(self):
        if not self.last_images:
            return
        folder = filedialog.askdirectory(title='Выбрать папку')
        if not folder:
            return
        base = os.path.splitext(os.path.basename(self.img_path))[0]
        for key, img in self.last_images.items():
            if key in ('Маска','Ядра'):
                pil = Image.fromarray(img)
            else:
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil.save(os.path.join(folder, f"{base}_{key}.png"))
        messagebox.showinfo('Сохранено', f'Сохранено в {folder}')

    def _show_img(self, title, cv_img, is_gray=False):
        if is_gray:
            pil = Image.fromarray(cv_img)
        else:
            pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        w,h = pil.size
        scale = min(400/w, 400/h, 1)
        if scale < 1:
            pil = pil.resize((int(w*scale), int(h*scale)), RESAMPLE)
        tk_img = ImageTk.PhotoImage(pil)
        lbl = self.labels[title]
        lbl.config(image=tk_img)
        lbl.image = tk_img

if __name__ == '__main__':
    app = CardsApp()
    app.mainloop()
