import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageTk
import os

try:
    from PIL.Image import Resampling
    RESAMPLE_METHOD = Resampling.LANCZOS
except ImportError:
    from PIL import Image
    RESAMPLE_METHOD = Image.ANTIALIAS

def imread_unicode(image_path):
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise FileNotFoundError(f"Не удалось загрузить изображение {image_path}. Ошибка: {e}")

def count_eggs(
    image_path,
    w_lower_r, w_lower_g, w_lower_b,
    w_upper_r, w_upper_g, w_upper_b,
    r_lower_l, r_lower_a, r_lower_b,
    r_upper_l, r_upper_a, r_upper_b,
    w_kernel_size, w_morph_iter, w_blur_ksize,
    r_kernel_size, r_morph_iter, r_blur_ksize,
    min_component_size,
    min_contour_area,
    max_contour_area,
    aspect_ratio_min,
    aspect_ratio_max,
    circularity_min,
    circularity_max
):

    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение {image_path}")
    h_img, w_img = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output = image_rgb.copy()

    def preprocessing(rgb_image):
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        kernel_small = np.ones((3, 3), np.uint8)
        l_dil = cv2.dilate(l, kernel_small, iterations=1)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_dil)
        l_blur = cv2.medianBlur(l_clahe, 5)
        l_processed = cv2.dilate(l_blur, kernel_small, iterations=2)
        lab_processed = cv2.merge((l_processed, a, b))
        return cv2.cvtColor(lab_processed, cv2.COLOR_LAB2RGB)

    preproc_rgb_white = preprocessing(image_rgb)
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    lower_white_rgb = np.array([w_lower_r, w_lower_g, w_lower_b], dtype='uint8')
    upper_white_rgb = np.array([w_upper_r, w_upper_g, w_upper_b], dtype='uint8')
    white_mask = cv2.inRange(preproc_rgb_white, lower_white_rgb, upper_white_rgb)

    lower_red_lab = np.array([r_lower_l, r_lower_a, r_lower_b], dtype='uint8')
    upper_red_lab = np.array([r_upper_l, r_upper_a, r_upper_b], dtype='uint8')
    red_mask = cv2.inRange(lab_image, lower_red_lab, upper_red_lab)

    if w_kernel_size < 1:
        w_kernel_size = 1
    kernel_w = np.ones((w_kernel_size, w_kernel_size), np.uint8)
    if w_morph_iter > 0:
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_w, iterations=w_morph_iter)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_w, iterations=w_morph_iter)
    if w_blur_ksize > 1:
        white_mask = cv2.blur(white_mask, (w_blur_ksize, w_blur_ksize))

    if r_kernel_size < 1:
        r_kernel_size = 1
    kernel_r = np.ones((r_kernel_size, r_kernel_size), np.uint8)
    if r_morph_iter > 0:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_r, iterations=r_morph_iter)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_r, iterations=r_morph_iter)
    if r_blur_ksize > 1:
        red_mask = cv2.blur(red_mask, (r_blur_ksize, r_blur_ksize))

    def remove_small_components(mask, min_size=5000):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        new_mask = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_size:
                new_mask[labels == lbl] = 255
        return new_mask

    white_mask = remove_small_components(white_mask, min_component_size)
    red_mask   = remove_small_components(red_mask,   min_component_size)

    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _   = cv2.findContours(red_mask,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_white_contours = [cnt for cnt in white_contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
    valid_red_contours   = [cnt for cnt in red_contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

    final_white_contours = []
    final_red_contours = []

    def check_egg_shape(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or (x + w) == w_img or (y + h) == h_img:
            return False
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        aspect = float(w) / float(h)
        circularity = 0
        if perimeter > 0:
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        return (aspect_ratio_min < aspect < aspect_ratio_max) and (circularity_min < circularity < circularity_max)

    for cnt in valid_white_contours:
        if check_egg_shape(cnt):
            final_white_contours.append(cnt)
    for cnt in valid_red_contours:
        if check_egg_shape(cnt):
            final_red_contours.append(cnt)

    cv2.drawContours(output, final_white_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(output, final_red_contours,   -1, (0, 255, 255), 2)

    white_count = len(final_white_contours)
    red_count = len(final_red_contours)
    total_count = white_count + red_count
    color_counts = {'белые': white_count, 'красные': red_count}

    images_dict = {
        "original": image_rgb,
        "white_mask": white_mask,
        "red_mask": red_mask,
        "result": output
    }
    return total_count, color_counts, images_dict


class EggDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа №1: Подсчет яиц")
        self.geometry("900x600")
        self.resizable(True, True)

        self.default_params = {
            "w_lower_r": 205, "w_lower_g": 215, "w_lower_b": 212,
            "w_upper_r": 255, "w_upper_g": 255, "w_upper_b": 255,
            "r_lower_l": 60,  "r_lower_a": 142, "r_lower_b": 110,
            "r_upper_l": 255,  "r_upper_a": 160, "r_upper_b": 255,
            "w_kernel_size": 5, "w_morph_iter": 1, "w_blur_ksize": 3,
            "r_kernel_size": 12, "r_morph_iter": 1, "r_blur_ksize": 1,
            "min_component_size": 5000,
            "min_contour_area": 13700,
            "max_contour_area": 999999,
            "aspect_ratio_min": 0.3,
            "aspect_ratio_max": 1.54,
            "circularity_min": 0.16,
            "circularity_max": 1.5
        }

        self.img_path = tk.StringVar(value="")

        self.w_lower_r = tk.IntVar(value=self.default_params["w_lower_r"])
        self.w_lower_g = tk.IntVar(value=self.default_params["w_lower_g"])
        self.w_lower_b = tk.IntVar(value=self.default_params["w_lower_b"])
        self.w_upper_r = tk.IntVar(value=self.default_params["w_upper_r"])
        self.w_upper_g = tk.IntVar(value=self.default_params["w_upper_g"])
        self.w_upper_b = tk.IntVar(value=self.default_params["w_upper_b"])
        self.w_kernel_size = tk.IntVar(value=self.default_params["w_kernel_size"])
        self.w_morph_iter  = tk.IntVar(value=self.default_params["w_morph_iter"])
        self.w_blur_ksize  = tk.IntVar(value=self.default_params["w_blur_ksize"])

        self.r_lower_l = tk.IntVar(value=self.default_params["r_lower_l"])
        self.r_lower_a = tk.IntVar(value=self.default_params["r_lower_a"])
        self.r_lower_b = tk.IntVar(value=self.default_params["r_lower_b"])
        self.r_upper_l = tk.IntVar(value=self.default_params["r_upper_l"])
        self.r_upper_a = tk.IntVar(value=self.default_params["r_upper_a"])
        self.r_upper_b = tk.IntVar(value=self.default_params["r_upper_b"])
        self.r_kernel_size = tk.IntVar(value=self.default_params["r_kernel_size"])
        self.r_morph_iter  = tk.IntVar(value=self.default_params["r_morph_iter"])
        self.r_blur_ksize  = tk.IntVar(value=self.default_params["r_blur_ksize"])

        self.min_component_size = tk.IntVar(value=self.default_params["min_component_size"])
        self.min_contour_area   = tk.IntVar(value=self.default_params["min_contour_area"])
        self.max_contour_area   = tk.IntVar(value=self.default_params["max_contour_area"])
        self.aspect_ratio_min   = tk.DoubleVar(value=self.default_params["aspect_ratio_min"])
        self.aspect_ratio_max   = tk.DoubleVar(value=self.default_params["aspect_ratio_max"])
        self.circularity_min    = tk.DoubleVar(value=self.default_params["circularity_min"])
        self.circularity_max    = tk.DoubleVar(value=self.default_params["circularity_max"])

        container = tk.Frame(self)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(container)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.scroll_canvas = tk.Canvas(left_frame, width=600)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.param_container = tk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.param_container, anchor="nw")
        self.param_container.bind("<Configure>", self.on_param_configure)

        right_frame = tk.Frame(container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        file_frame = tk.Frame(self.param_container)
        file_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        btn_select = tk.Button(file_frame, text="Выбрать изображение", command=self.select_image)
        btn_select.pack(side=tk.LEFT)

        w_frame = tk.LabelFrame(self.param_container, text="Белые яйца")
        w_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        self._add_scale(w_frame, "w_lower_r", "W Lower R", self.w_lower_r, 0, 255, 1)
        self._add_scale(w_frame, "w_lower_g", "W Lower G", self.w_lower_g, 0, 255, 1)
        self._add_scale(w_frame, "w_lower_b", "W Lower B", self.w_lower_b, 0, 255, 1)
        self._add_scale(w_frame, "w_upper_r", "W Upper R", self.w_upper_r, 0, 255, 1)
        self._add_scale(w_frame, "w_upper_g", "W Upper G", self.w_upper_g, 0, 255, 1)
        self._add_scale(w_frame, "w_upper_b", "W Upper B", self.w_upper_b, 0, 255, 1)
        self._add_scale(w_frame, "w_kernel_size", "kernel_size", self.w_kernel_size, 1, 20, 1)
        self._add_scale(w_frame, "w_morph_iter",  "morph_iter",  self.w_morph_iter, 0, 5, 1)
        self._add_scale(w_frame, "w_blur_ksize",  "blur_ksize",  self.w_blur_ksize, 1, 20, 1)

        r_frame = tk.LabelFrame(self.param_container, text="Красные яйца (LAB)")
        r_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nw")
        self._add_scale(r_frame, "r_lower_l", "R Lower L", self.r_lower_l, 0, 255, 1)
        self._add_scale(r_frame, "r_lower_a", "R Lower a", self.r_lower_a, 0, 255, 1)
        self._add_scale(r_frame, "r_lower_b", "R Lower b", self.r_lower_b, 0, 255, 1)
        self._add_scale(r_frame, "r_upper_l", "R Upper L", self.r_upper_l, 0, 255, 1)
        self._add_scale(r_frame, "r_upper_a", "R Upper a", self.r_upper_a, 0, 255, 1)
        self._add_scale(r_frame, "r_upper_b", "R Upper b", self.r_upper_b, 0, 255, 1)
        self._add_scale(r_frame, "r_kernel_size", "kernel_size", self.r_kernel_size, 1, 20, 1)
        self._add_scale(r_frame, "r_morph_iter",  "morph_iter",  self.r_morph_iter, 0, 5, 1)
        self._add_scale(r_frame, "r_blur_ksize",  "blur_ksize",  self.r_blur_ksize, 1, 20, 1)

        other_frame = tk.LabelFrame(self.param_container, text="Прочие параметры")
        other_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nw")
        self._add_scale(other_frame, "min_component_size", "min_component_size", self.min_component_size, 0, 30000, 100)
        self._add_scale(other_frame, "min_contour_area",   "min_contour_area",   self.min_contour_area,   0, 100000, 100)
        self._add_scale(other_frame, "max_contour_area",   "max_contour_area",   self.max_contour_area,   1000, 2000000, 100)
        tk.Label(other_frame, text="aspect_ratio_min").pack(anchor="w")
        tk.Spinbox(other_frame, from_=0.0, to=5.0, increment=0.01, textvariable=self.aspect_ratio_min).pack(anchor="w", fill=tk.X)
        tk.Label(other_frame, text="aspect_ratio_max").pack(anchor="w")
        tk.Spinbox(other_frame, from_=0.0, to=5.0, increment=0.01, textvariable=self.aspect_ratio_max).pack(anchor="w", fill=tk.X)
        tk.Label(other_frame, text="circularity_min").pack(anchor="w")
        tk.Spinbox(other_frame, from_=0.0, to=3.0, increment=0.01, textvariable=self.circularity_min).pack(anchor="w", fill=tk.X)
        tk.Label(other_frame, text="circularity_max").pack(anchor="w")
        tk.Spinbox(other_frame, from_=0.0, to=3.0, increment=0.01, textvariable=self.circularity_max).pack(anchor="w", fill=tk.X)
        btn_frame = tk.Frame(other_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        btn_defaults = tk.Button(btn_frame, text="Сбросить на дефолт", command=self.reset_defaults)
        btn_defaults.pack(side=tk.LEFT, fill=tk.X, expand=True)
        btn_process = tk.Button(btn_frame, text="Обработать", command=self.process_image)
        btn_process.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stats_frame = tk.LabelFrame(other_frame, text="Статистика")
        self.stats_frame.pack(fill=tk.X, pady=5)
        self.stats_label = tk.Label(self.stats_frame, text="", justify="left")
        self.stats_label.pack(padx=5, pady=5, fill=tk.X)

        self.display_frame = tk.LabelFrame(right_frame, text="Результаты обработки")
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        self.label_original = tk.Label(self.display_frame, text="Исходное изображение")
        self.label_original.grid(row=0, column=0, padx=5, pady=5)
        self.label_white_mask = tk.Label(self.display_frame, text="Маска белых")
        self.label_white_mask.grid(row=0, column=1, padx=5, pady=5)
        self.label_red_mask = tk.Label(self.display_frame, text="Маска красных")
        self.label_red_mask.grid(row=1, column=0, padx=5, pady=5)
        self.label_result = tk.Label(self.display_frame, text="Результат")
        self.label_result.grid(row=1, column=1, padx=5, pady=5)
        self.save_button = tk.Button(self.display_frame, text="Сохранить результат", command=self.save_results, state="disabled")
        self.save_button.grid(row=2, column=0, columnspan=2, pady=10)
        self.last_images = {}

    def _add_scale(self, parent, var_name, label_text, tk_var, min_val, max_val, resolution_step):
        tk.Label(parent, text=label_text).pack(anchor="w")
        scale = tk.Scale(parent, from_=min_val, to=max_val, resolution=resolution_step, orient=tk.HORIZONTAL, variable=tk_var)
        scale.pack(fill=tk.X)

    def on_param_configure(self, event):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def select_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=filetypes)
        if path:
            self.img_path.set(path)
            try:
                cv_img = imread_unicode(path)
                if cv_img is None:
                    raise FileNotFoundError(f"Не удалось загрузить изображение {path}")
                cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                pil_img = self.cv2_to_pil(cv_img_rgb)
                self.show_image_on_label(self.label_original, pil_img)
            except Exception as e:
                self.stats_label.config(text=f"Ошибка загрузки: {e}")
            self.update_idletasks()
            self.geometry("")

    def reset_defaults(self):
        for k, v in self.default_params.items():
            if hasattr(self, k):
                getattr(self, k).set(v)

    def process_image(self):
        if not self.img_path.get():
            self.stats_label.config(text="Сначала выберите изображение.")
            return
        try:
            total_count, color_counts, images_dict = count_eggs(
                image_path=self.img_path.get(),
                w_lower_r=self.w_lower_r.get(),
                w_lower_g=self.w_lower_g.get(),
                w_lower_b=self.w_lower_b.get(),
                w_upper_r=self.w_upper_r.get(),
                w_upper_g=self.w_upper_g.get(),
                w_upper_b=self.w_upper_b.get(),
                r_lower_l=self.r_lower_l.get(),
                r_lower_a=self.r_lower_a.get(),
                r_lower_b=self.r_lower_b.get(),
                r_upper_l=self.r_upper_l.get(),
                r_upper_a=self.r_upper_a.get(),
                r_upper_b=self.r_upper_b.get(),
                w_kernel_size=self.w_kernel_size.get(),
                w_morph_iter=self.w_morph_iter.get(),
                w_blur_ksize=self.w_blur_ksize.get(),
                r_kernel_size=self.r_kernel_size.get(),
                r_morph_iter=self.r_morph_iter.get(),
                r_blur_ksize=self.r_blur_ksize.get(),
                min_component_size=self.min_component_size.get(),
                min_contour_area=self.min_contour_area.get(),
                max_contour_area=self.max_contour_area.get(),
                aspect_ratio_min=self.aspect_ratio_min.get(),
                aspect_ratio_max=self.aspect_ratio_max.get(),
                circularity_min=self.circularity_min.get(),
                circularity_max=self.circularity_max.get()
            )
        except FileNotFoundError as e:
            self.stats_label.config(text=str(e))
            return
        except Exception as e:
            self.stats_label.config(text=f"Ошибка: {e}")
            return
        stats_text = (f"Всего яиц: {total_count}\n"
                      f"Белые: {color_counts['белые']}\n"
                      f"Красные: {color_counts['красные']}\n")
        self.stats_label.config(text=stats_text)
        self.last_images = images_dict
        self.last_images["original_pil"] = self.cv2_to_pil(images_dict["original"])
        self.last_images["white_mask_pil"] = self.cv2_to_pil(images_dict["white_mask"], is_mask=True)
        self.last_images["red_mask_pil"] = self.cv2_to_pil(images_dict["red_mask"], is_mask=True)
        self.last_images["result_pil"] = self.cv2_to_pil(images_dict["result"])
        self.show_image_on_label(self.label_original, self.last_images["original_pil"])
        self.show_image_on_label(self.label_white_mask, self.last_images["white_mask_pil"])
        self.show_image_on_label(self.label_red_mask,   self.last_images["red_mask_pil"])
        self.show_image_on_label(self.label_result,     self.last_images["result_pil"])
        self.save_button.config(state="normal")
        self.update_idletasks()
        self.geometry("")

    def save_results(self):
        if not self.last_images:
            return
        folder_selected = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if not folder_selected:
            return
        base_name = os.path.splitext(os.path.basename(self.img_path.get()))[0]
        paths = []
        for key in ["original_pil", "white_mask_pil", "red_mask_pil", "result_pil"]:
            out_path = os.path.join(folder_selected, f"{base_name}_{key}.png")
            self.last_images[key].save(out_path)
            paths.append(out_path)
        self.stats_label.config(text="Результаты сохранены в:\n" + "\n".join(paths))
        self.update_idletasks()

    def cv2_to_pil(self, cv_img, is_mask=False):
        if is_mask:
            if len(cv_img.shape) == 2:
                pil_img = Image.fromarray(cv_img, mode='L')
            else:
                pil_img = Image.fromarray(cv_img[..., 0], mode='L')
        else:
            if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                pil_img = Image.fromarray(cv_img)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        return pil_img

    def show_image_on_label(self, label, pil_image):
        max_w, max_h = 350, 350
        w, h = pil_image.size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            pil_image = pil_image.resize(new_size, RESAMPLE_METHOD)
        tk_img = ImageTk.PhotoImage(pil_image)
        label.config(image=tk_img)
        label.image = tk_img


if __name__ == "__main__":
    app = EggDetectorApp()
    app.mainloop()
