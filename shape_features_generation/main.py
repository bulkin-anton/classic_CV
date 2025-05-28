import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from scipy.spatial import ConvexHull


def read_tif_image(path: Path) -> np.ndarray:
    path_str = str(path)
    try:
        with open(path_str, "rb") as f:
            img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Cannot decode file {path_str}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise FileNotFoundError(f'Cannot read file {path_str}: {e}')

def kmeans_segmentation(img_rgb: np.ndarray, n_clusters: int = 2):
    h, w = img_rgb.shape[:2]
    data = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten().reshape(h, w)
    centers = centers.astype(np.uint8)
    return labels, centers

def mask_brightest_cluster(img_rgb: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v_channel = img_hsv[..., 2]
    means = []
    for i in range(n_clusters):
        mask = (labels == i)
        means.append(v_channel[mask].mean() if np.any(mask) else 0)
    cluster = int(np.argmax(means))
    mask = (labels == cluster).astype(np.uint8) * 255
    return mask

def choose_main_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return mask
    max_idx = 1 + np.argmax(areas)
    return ((labels == max_idx) * 255).astype(np.uint8)

def segment_palm_kmeans(img_rgb: np.ndarray, params=None) -> np.ndarray:
    if params is None:
        params = dict(erode1=1, erode2=2, dilate1=1, erode3=3, erode4=1, kernel_size=5)
    n_clusters = 2
    labels, centers = kmeans_segmentation(img_rgb, n_clusters=n_clusters)
    palm_mask_raw = mask_brightest_cluster(img_rgb, labels, n_clusters=n_clusters)
    kernel = np.ones((params['kernel_size'], params['kernel_size']), np.uint8)
    mask = cv2.erode(palm_mask_raw, np.ones((7, 7), np.uint8), iterations=params['erode1'])
    mask = cv2.erode(mask, kernel, iterations=params['erode2'])
    mask = cv2.dilate(mask, kernel, iterations=params['dilate1'])
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=params['erode3'])
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=params['erode4'])
    palm_mask = choose_main_component(mask)
    return palm_mask

def skeleton_graph(mask: np.ndarray):
    skel = skeletonize(mask > 0).astype(np.uint8)
    h, w = skel.shape
    G = nx.Graph()
    pts = np.argwhere(skel)
    for y, x in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy, dx) == (0, 0):
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
                    G.add_edge((y, x), (ny, nx_))
    cy, cx = np.mean(np.argwhere(mask > 0), axis=0)
    return G, (int(cx), int(cy)), skel

def find_top3_tips(tips, center):
    dists = [np.linalg.norm(np.array(pt) - center) for pt in tips]
    top3_idx = np.argsort(dists)[-3:]
    return [tips[i] for i in top3_idx]

def compute_fingers_direction(top3, center):
    top3 = np.array(top3)
    mean_tip = np.mean(top3, axis=0)
    direction = mean_tip - center
    direction = direction / np.linalg.norm(direction)
    return direction

def filter_tips_by_direction(tips, center, direction, angle_threshold=np.pi/2):
    filtered = []
    for pt in tips:
        v = np.array(pt) - center
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        angle = np.arccos(np.clip(np.dot(v_norm, direction), -1.0, 1.0))
        if angle < angle_threshold:
            filtered.append(tuple(pt))
    return filtered

def five_tips(G, center):
    all_tips = [(x, y) for y, x in G.nodes if G.degree((y, x)) == 1]
    if len(all_tips) < 3:
        return all_tips
    top3 = find_top3_tips(all_tips, np.array(center))
    direction = compute_fingers_direction(top3, np.array(center))
    tips_filtered = filter_tips_by_direction(all_tips, np.array(center), direction, angle_threshold=np.deg2rad(120))
    if len(tips_filtered) > 5:
        tips_filtered = sorted(tips_filtered, key=lambda pt: -np.linalg.norm(np.array(pt) - center))[:5]
    elif len(tips_filtered) < 5:
        rest = [pt for pt in all_tips if pt not in tips_filtered]
        rest = sorted(rest, key=lambda pt: -np.linalg.norm(np.array(pt) - center))
        tips_filtered += rest[:(5 - len(tips_filtered))]
    return tips_filtered[:5]

def get_finger_angles(tips, center):
    tips = np.array(tips)
    center = np.array(center)
    return np.arctan2(tips[:,1] - center[1], tips[:,0] - center[0])

def get_thumb_index(tips, center):
    angles = get_finger_angles(tips, center)
    idx_min = np.argmin(angles)
    idx_max = np.argmax(angles)
    mean_angle = np.mean(np.delete(angles, [idx_min, idx_max]))
    if abs(angles[idx_min] - mean_angle) > abs(angles[idx_max] - mean_angle):
        return idx_min
    else:
        return idx_max

def sort_fingers_robust(tips, center):
    tips = np.array(tips)
    if len(tips) != 5:
        angles = get_finger_angles(tips, center)
        sort_idx = np.argsort(angles)
        return [tuple(tips[i]) for i in sort_idx]
    hull = ConvexHull(tips)
    hull_order = hull.vertices
    thumb_idx_in_tips = get_thumb_index(tips, center)
    thumb_in_hull = np.where(hull_order == thumb_idx_in_tips)[0][0]
    ordered = [tuple(tips[hull_order[(i + thumb_in_hull) % 5]]) for i in range(5)]
    return [ordered[0]] + ordered[1:][::-1]

def number_and_pose(tips, center, thresholds):
    tips_sorted = sort_fingers_robust(tips, center)
    d = [np.linalg.norm(np.array(tips_sorted[i]) - np.array(tips_sorted[(i + 1) % 5]))
         for i in range(5)]
    d_pairs = d[:4]
    pose = ''.join(['+' if d_i < thr else '-' for d_i, thr in zip(d_pairs, thresholds)])
    code = '1' + pose[0] + '2' + pose[1] + '3' + pose[2] + '4' + pose[3] + '5'
    return tips_sorted, code

def visualize_for_gui(img, tips, pose_code):
    img_vis = img.copy()
    for i, pt in enumerate(tips, 1):
        cv2.circle(img_vis, pt, 10, (255, 0, 0), -1)
        offset = (0, -25)
        text_pos = (pt[0] + offset[0], pt[1] + offset[1])
        cv2.putText(img_vis, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(img_vis, pose_code, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    return img_vis

def visualize_skeleton(mask):
    bin_mask = mask.astype(bool)
    skel = skeletonize(bin_mask)
    left = np.dstack([bin_mask.astype(np.uint8) * 255] * 3)
    right = left.copy()
    right[skel, 0] = 255
    right[skel, 1] = 0
    right[skel, 2] = 0
    combined = right
    return combined

def save_results_single(out_dir, img_name, orig, mask_skel, processed, tips, pose_code):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    base = Path(img_name).stem
    cv2.imwrite(str(out_dir / f"{base}_original.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base}_mask_skeleton.png"), cv2.cvtColor(mask_skel, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base}_processed.png"), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    with open(out_dir / f"results_{base}.txt", 'w', encoding='utf-8') as f:
        coords_str = " ".join([f"{pt[0]} {pt[1]}" for pt in tips])
        f.write(f"{coords_str} {pose_code}\n")

def save_results_batch(out_dir, file_pose_tips_list):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / "Results.txt", 'w', encoding='utf-8') as f:
        for fname, pose, tips in file_pose_tips_list:
            coords_str = " ".join([f"{pt[0]} {pt[1]}" for pt in tips])
            f.write(f"{fname} {coords_str} {pose}\n")

class PalmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка ладони")
        self.img = None
        self.result_img = None
        self.mask_skel_img = None
        self.img_path = None
        self.last_pose_code = None
        self.last_tips = None

        self.params = {
            'erode1': tk.IntVar(value=1),
            'erode2': tk.IntVar(value=2),
            'dilate1': tk.IntVar(value=1),
            'erode3': tk.IntVar(value=3),
            'erode4': tk.IntVar(value=1),
            'kernel_size': tk.IntVar(value=5)
        }
        self.thresholds = [167.48, 62.63, 52.59, 84.83]

        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10)
        tk.Button(frame, text="Загрузить изображение", command=self.open_image).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="Обработать", command=self.process_image).grid(row=0, column=1, padx=5)
        tk.Button(frame, text="Сохранить результат", command=self.save_current).grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Пакетная обработка", command=self.batch_process).grid(row=0, column=3, padx=5)
        tk.Button(frame, text="Дефолтные морф. параметры", command=self.set_default_params).grid(row=0, column=4, padx=5)

        row = 1
        for i, (param, var) in enumerate(self.params.items()):
            tk.Label(frame, text=param).grid(row=row, column=i)
            tk.Scale(frame, from_=1, to=10, orient='horizontal', variable=var, length=80).grid(row=row+1, column=i)

        self.canvas1 = tk.Label(root)
        self.canvas1.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas2 = tk.Label(root)
        self.canvas2.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas3 = tk.Label(root)
        self.canvas3.pack(side=tk.LEFT, padx=10, pady=10)

    def set_default_params(self):
        defaults = dict(erode1=1, erode2=2, dilate1=1, erode3=3, erode4=1, kernel_size=5)
        for k, v in defaults.items():
            self.params[k].set(v)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.tif *.png *.jpg *.bmp")])
        if not path:
            return
        self.img_path = Path(path)
        self.img = read_tif_image(self.img_path)
        self.show_image(self.img, self.canvas1)

    def get_current_params(self):
        return {k: v.get() for k, v in self.params.items()}

    def process_image(self):
        if self.img is None:
            messagebox.showinfo("Ошибка", "Сначала загрузите изображение")
            return
        params = self.get_current_params()
        mask = segment_palm_kmeans(self.img, params)
        G, center, skel = skeleton_graph(mask)
        mask_skel_vis = visualize_skeleton(mask)
        self.mask_skel_img = mask_skel_vis
        tips = five_tips(G, center)
        if len(tips) < 5:
            messagebox.showinfo("Ошибка", "Найдено меньше 5 кончиков пальцев")
            return
        tips_sorted, pose_code = number_and_pose(tips, center, self.thresholds)
        self.result_img = visualize_for_gui(self.img, tips_sorted, pose_code)
        self.show_image(self.mask_skel_img, self.canvas2)
        self.show_image(self.result_img, self.canvas3)
        self.last_tips = tips_sorted
        self.last_pose_code = pose_code

    def show_image(self, img, canvas):
        im = Image.fromarray(img).resize((510, 702), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(im)
        canvas.imgtk = tk_img
        canvas.configure(image=tk_img)

    def save_current(self):
        if self.img is None or self.result_img is None or self.mask_skel_img is None:
            messagebox.showinfo("Ошибка", "Сначала загрузите и обработайте изображение")
            return
        out_dir = filedialog.askdirectory(title="Выберите папку для сохранения")
        if not out_dir:
            return
        img_name = self.img_path.name
        save_results_single(out_dir, img_name, self.img, self.mask_skel_img, self.result_img, self.last_tips, self.last_pose_code)
        messagebox.showinfo("Сохранено", f"Файлы сохранены в {out_dir}")

    def batch_process(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.tif *.png *.jpg *.bmp")])
        if not paths:
            return
        out_dir = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if not out_dir:
            return
        params = self.get_current_params()
        results = []
        for path in paths:
            img = read_tif_image(Path(path))
            mask = segment_palm_kmeans(img, params)
            G, center, skel = skeleton_graph(mask)
            mask_skel_vis = visualize_skeleton(img, mask, skel)
            tips = five_tips(G, center)
            if len(tips) < 5:
                continue
            tips_sorted, pose_code = number_and_pose(tips, center, self.thresholds)
            processed = visualize_for_gui(img, tips_sorted, pose_code)
            save_results_single(out_dir, Path(path).name, img, mask_skel_vis, processed, tips_sorted, pose_code)
            results.append((Path(path).name, pose_code, tips_sorted))
        save_results_batch(out_dir, results)
        messagebox.showinfo("Готово", f"Обработано файлов: {len(results)}.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PalmApp(root)
    root.mainloop()
