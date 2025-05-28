# Classic_Computer_Vision

A set of classic computer vision tasks with implementation, experimental research, and detailed reports.  
Each module is an independent project that includes a task statement, Python code, and a PDF report analyzing the results.

---

## 📂 Repository structure

- [image_processing_segmentation/](image_processing_segmentation/)  
  Image segmentation tasks:
1. **main.py** is a GUI application on Tkinter for uploading images and performing segmentation using OpenCV.  
  2. **task.pdf** — statement of the assignment and requirements.  
  3. **report_image_segmentation.pdf** — a report with theory, description of the algorithm and examples of results.

- [object_form_analysis/](object_form_analysis/)  
  Exploring the shapes of objects in images:
1. **main.py** — a script for contour extraction and analysis of geometric properties (area, perimeter, eccentricity).  
  2. **task.pdf** — setting a task for analyzing forms.  
  3. **report_object_form_analysis.pdf** — a report with methodology, implementation and illustrations.

- [shape_features_generation/](shape_features_generation/)  
  Generating and visualizing shape descriptors:
  1. **main.py** is a code for constructing skeletonization, convex hulls, and other shape features.  
  2. **task.pdf** — description of required features and metrics.  
  3. **report_shape_features_generation.pdf** — detailed analysis of methods and obtained features.

---

## 🛠️ Technologies used

- **Language**: Python 3  
- **Libraries**:
- OpenCV (`cv2`) — basic image operations, contours, morphology.  
  - NumPy — working with arrays of pixels.  
  - Pillow (PIL) — loading and saving images.  
  - Matplotlib — plotting and visualization of intermediate results.  
  - scikit-image — auxiliary functions (skeletonization, filtering).  
  - Tkinter is a simple graphical shell for demo applications.  

---

> **Each practical assignment contains detailed theoretical background, a Python implementation, and demonstration examples that showcase the methods in practice.**
