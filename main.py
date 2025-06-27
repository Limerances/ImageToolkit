import sys
import os
from functools import partial

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QLabel,
    QAction,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QMenu,
    QToolBar,
    QScrollArea,
)

from process import ImageProcessor

def np_to_qpix(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        h, w = img.shape
        fmt = QImage.Format_Grayscale8
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, fmt)
    else:
        h, w, ch = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def read_raw(path: str, width: int, height: int, channels: int = 1) -> np.ndarray:
    dtype = np.uint8
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    if channels == 1:
        img = data.reshape((height, width))
    else:
        img = data.reshape((height, width, channels))
    return img

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    import numpy as np
    import cv2
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception:
        return None

class ImageSubWindow(QMdiSubWindow):
    def __init__(self, img: np.ndarray, title: str):
        super().__init__()
        self.img = img.copy() 
        self.pix_orig = np_to_qpix(self.img)
        self.label = QLabel(alignment=Qt.AlignCenter)
        self.setWidget(self.label)
        self.setWindowTitle(title)
        self.rescale() 

    def rescale(self):
        rect = self.contentsRect().size()
        if rect.isEmpty():
            return
        scaled = self.pix_orig.scaled(
            rect, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.rescale()

    def apply(self, func, *args, **kwargs):
        result = func(self.img, *args, **kwargs)
        self.img = result
        self.pix_orig = np_to_qpix(self.img)
        self.rescale()
        
    def update(self, img):
        self.img = img
        self.pix_orig = np_to_qpix(self.img)
        self.rescale()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小狐的图像处理工具集")
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.create_actions()
        self.create_menus()
        self.create_toolbar()
        
        screen_geo = QGuiApplication.primaryScreen().availableGeometry()
        sw, sh = screen_geo.width(), screen_geo.height()
        ratio_main = 0.8
        self.resize(int(sw * ratio_main), int(sh * ratio_main))
        
        self.mdi.subWindowActivated.connect(self.highlight_sub)

    def create_menus(self):
        bar = self.menuBar()
        
        # bar.setStyleSheet("""
        #     QMenuBar {
        #         background-color: #2c3e50;
        #         color: white;
        #         font: 14px "Microsoft YaHei";
        #     }

        #     QMenuBar::item {
        #         background: transparent;
        #         padding: 6px 15px;
        #     }

        #     QMenuBar::item:selected {
        #         background: #34495e;
        #     }

        #     QMenu {
        #         background-color: #ecf0f1;
        #         color: #2c3e50;
        #     }

        #     QMenu::item:selected {
        #         background-color: #3498db;
        #         color: white;
        #     }
        #     """)
        
        bar.setStyleSheet("""
            QMenuBar {
                background-color: #2c3e50;
                color: white;
                font: bold 18px "Microsoft YaHei";
            }

            QMenuBar::item {
                padding: 10px 20px;
            }

            QMenuBar::item:selected {
                background-color: #34495e;
            }

            QMenu {
                background-color: #ecf0f1;
                font: 16px "Microsoft YaHei";
            }

            QMenu::item {
                padding: 8px 20px;
            }

            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
            """)

        
        
        file_menu = bar.addMenu("文件")
        file_menu.addAction(self.openAct)
        file_menu.addAction(self.saveAct)

        basic_menu = bar.addMenu("基本操作")
        basic_menu.addAction(self.addAct)
        basic_menu.addAction(self.invAct)
        basic_menu.addAction(self.transAct)
        basic_menu.addAction(self.rotAct)
        basic_menu.addAction(self.mirrorAct)

        fft_menu = bar.addMenu("傅里叶变换")
        fft_menu.addAction(self.fftAct)
        fft_menu.addAction(self.ifftAct)
        fft_menu.addAction(self.fftReverseAct)

        enhance_menu = bar.addMenu("图像增强")
        enhance_menu.addAction(self.histEqAct)
        enhance_menu.addAction(self.homoAct)
        enhance_menu.addAction(self.expAct)
        enhance_menu.addAction(self.laplaceAct)
        
        edge_menu = bar.addMenu("边缘检测")
        edge_menu.addAction(self.robertsAct)
        edge_menu.addAction(self.prewittAct)
        edge_menu.addAction(self.sobelAct)
        edge_menu.addAction(self.lapEdgeAct)
        
        fd_menu = bar.addMenu("傅里叶描述子")
        fd_menu.addAction(self.fdAct)
        
        morph_menu = bar.addMenu("膨胀与腐蚀")
        morph_menu.addAction(self.dilateOnceAct)
        morph_menu.addAction(self.erodeOnceAct)
        morph_menu.addAction(self.dilateRepeatAct)
        morph_menu.addAction(self.erodeRepeatAct)
        
        match_menu = bar.addMenu("特征匹配")
        match_menu.addAction(self.siftMatchAct)
        
        info_menu = bar.addMenu("关于")
        info_menu.addAction(self.show_info)
        
    def create_actions(self):
        self.openAct = QAction("打开图片", self, triggered=self.open_image)
        self.saveAct = QAction("保存为BMP", self, triggered=self.save_as_bmp)
        self.test = QAction("test", self, triggered=self.test)
        self.re_open = QAction("打开默认图片", self, triggered=self.re_open)
        #基本操作
        self.addAct = QAction("图像叠加", self, triggered=self.add_images)
        self.invAct = QAction("图像取反", self, triggered=lambda: self.apply_current(ImageProcessor.invert))
        self.transAct = QAction("图像平移", self, triggered=self.translate_current)
        self.rotAct = QAction("图形旋转", self, triggered=self.rotate_current)
        self.mirrorAct = QAction("图形镜像", self, triggered=self.mirror_current)
        #FFT
        self.fftAct = QAction("FFT", self, triggered=self.fft2_current)
        self.ifftAct = QAction("FFT反变换", self, triggered=self.ifft2_current)
        self.fftReverseAct = QAction("进行(-1)^(x+y)变换", self, triggered=lambda: self.apply_current(ImageProcessor.conjugate_dft_inverse))
        #图像增强
        self.histEqAct = QAction("直方图均衡化", self, triggered=self.hist_eq_current)
        self.homoAct = QAction("同态滤波", self, triggered=self.homomorphic_current)
        self.expAct = QAction("指数增强", self, triggered=self.exp_current)
        self.laplaceAct= QAction("Laplace 锐化", self, triggered=self.laplace_current)
        #边缘检测
        self.robertsAct  = QAction("Roberts",  self, triggered=self.edge_roberts_current)
        self.prewittAct  = QAction("Prewitt",  self, triggered=self.edge_prewitt_current)
        self.sobelAct    = QAction("Sobel",    self, triggered=self.edge_sobel_current)
        self.lapEdgeAct  = QAction("Laplacian",self, triggered=self.edge_laplace_current)
        #傅里叶描述子
        self.fdAct = QAction("傅里叶重构", self, triggered=self.fd_current)
        #腐蚀与膨胀
        self.dilateOnceAct = QAction("一次膨胀", self, triggered=self.dilate_once_current)
        self.erodeOnceAct = QAction("一次腐蚀", self, triggered=self.erode_once_current)
        self.dilateRepeatAct = QAction("反复膨胀", self, triggered=self.dilate_repeated_current)
        self.erodeRepeatAct = QAction("反复腐蚀", self, triggered=self.erode_repeated_current)
        #SIFT匹配
        self.siftMatchAct = QAction("SIFT匹配", self, triggered=self.sift_match_current)
        #关于
        self.show_info = QAction("关于", self, triggered=self.show_info)
        

    def create_toolbar(self):
        tb = QToolBar("Main Toolbar")

        tb.setStyleSheet("""
            QToolBar {
                background-color: #333842;
                spacing: 12px;
                padding: 8px;
                color: white;
                font: bold 18px "Microsoft YaHei";
            }

            QToolButton {
                font: 16px "Microsoft YaHei";
                color: white;
                margin: 4px;
            }

            QToolButton:hover {
                background: #1B2B2E;
                border-radius: 6px;
            }
            """)


        
        tb.setIconSize(QSize(32, 32))
        self.addToolBar(tb)
        for act in [self.openAct, self.re_open]:
            tb.addAction(act)

    def re_open(self):
        #delete all subwindows
        for sub in self.mdi.subWindowList():
            self.mdi.removeSubWindow(sub)
            sub.deleteLater()
       
        # self.show_image(self.get_image(r"image\3.jpg")[0],r"3.jpg")
        # self.show_image(self.get_image(r"image\black_image_10.bmp")[0],r"black_image_10.bmp")
        # self.show_image(self.get_image(r"image\5.png")[0],r"5.png")
        self.show_image(self.get_image(resource_path(r"image\3.jpg"))[0], r"3.jpg")
        self.show_image(self.get_image(resource_path(r"image\black_image_10.bmp"))[0], r"black_image_10.bmp")
        self.show_image(self.get_image(resource_path(r"image\5.png"))[0], r"5.png")
            

    def show_info(self):
        QMessageBox.information(self, "关于", "图像处理工具集\n版本 1.0\n作者: ZY2406441\n2025年6月")

    def test(self):
        image = np.zeros((128, 128), dtype=np.uint8)
        size = 4
        min_index = 127 // 2 - size // 2 + 1
        max_index = min_index + size
        # image[127//2 - 1:127//2 + 3, 127//2 - 1:127//2 + 3] = 255
        # image[127//2:127//2 + 2, 127//2:127//2 + 2] = 255
        image[min_index:max_index, min_index:max_index] = 255
        # cv2.imwrite(f"image\\black_image_{size}.bmp", image)
        cv2.imwrite(resource_path(f"image\\black_image_{size}.bmp"), image)
        QMessageBox.information(self, "Test", f"128x128黑色图像已创建并保存为black_image_{size}.bmp")

    def active_sub(self):
        active = self.mdi.activeSubWindow()
        return active

    def get_image(self,path=None):
        
        if path is None:
            fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.bmp *.jpg *.png *.jpeg *.raw)")
            if not fname:
                return None, None
        else:
            fname = path

        img = None
        if fname.lower().endswith(".raw"):
            width, ok = QInputDialog.getInt(self, "RAW Width", "Width:")
            if not ok:
                return None, None
            height, ok = QInputDialog.getInt(self, "RAW Height", "Height:")
            if not ok:
                return None, None
            img = read_raw(fname, width, height)
        else:
            # img = cv2.imread(fname, cv2.IMREAD_COLOR)
            img = imread_unicode(fname) 
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image")
            return None, None
        
        return img, os.path.basename(fname)
    
    def show_image(self, img: np.ndarray, title: str,w_max_ratio=1,h_max_ratio=1):
        sub = ImageSubWindow(img, title)
        mdi_sub = self.mdi.addSubWindow(sub)
        mdi_sub.setAttribute(Qt.WA_DeleteOnClose)

        mw_w, mw_h = self.size().width(), self.size().height()
        ratio_sub = 0.4
        mdi_sub.resize(int(mw_w * ratio_sub), int(mw_h * ratio_sub))
        mdi_sub.setMinimumSize(int(mw_h * ratio_sub * 0.7), int(mw_h * ratio_sub * 0.7))
        mdi_sub.setMaximumSize(int(mw_h * ratio_sub * w_max_ratio), int(mw_h * ratio_sub * h_max_ratio))

        mdi_sub.show()
        return mdi_sub

    def open_image(self):
        img, title = self.get_image()
        if img is None:
            return
        self.show_image(img, title)

    def save_as_bmp(self):
        try:
            sub = self.active_sub()
            if not sub:
                return
            fname, _ = QFileDialog.getSaveFileName(self, "保存为BMP", "", "BMP Files (*.bmp)")
            if not fname:
                return
            cv2.imwrite(fname, sub.img)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"保存失败: {str(e)}")
            return
        QMessageBox.information(self, "保存成功", f"图像已保存为 {fname}")

    def apply_current(self, func, *args, **kwargs):
        sub = self.active_sub()
        if not sub:
            return
        sub.apply(func, *args, **kwargs)

    def add_images(self):
        sub = self.active_sub()
        if not sub:
            QMessageBox.warning(self, "Add", "请先选中一个图像窗口")
            return

        img1 = sub.img
        
        img2, fname = self.get_image()
        if img2 is None:
            return
        self.show_image(img2, fname)

        img_add = ImageProcessor.add(img1, img2)
        title_add = f"{sub.windowTitle()} + {os.path.basename(fname)}"
        self.show_image(img_add, title_add)

    def translate_current(self):
        dx, ok = QInputDialog.getInt(self, "X方向平移", "dx:")
        if not ok:
            return
        dy, ok = QInputDialog.getInt(self, "Y方向平移", "dy:")
        if not ok:
            return
        self.apply_current(ImageProcessor.translate, dx, dy)

    def rotate_current(self):
        angle, ok = QInputDialog.getDouble(self, "Rotate", "angle:")
        if not ok:
            return
        self.apply_current(ImageProcessor.rotate, angle)
        
    def mirror_current(self):
        axis, ok = QInputDialog.getItem(self, "图像镜像", "镜像轴:", ["水平镜像", "垂直镜像"], 0, False)
        if not ok:
            return
        if axis == "水平镜像":
            self.apply_current(ImageProcessor.mirror, axis=1)
        else:
            self.apply_current(ImageProcessor.mirror, axis=0)
            
    def fft2_current(self):
        sub = self.active_sub()
        if not sub:
            return
        disp1, disp2, disp3, fshift = ImageProcessor.fft2(sub.img)
        
        sub1 = self.show_image(disp1, f"Step1: 2D傅里叶变换")
        sub1.fft_data = fshift
        sub2 = self.show_image(disp2, f"Step2: 移动低频分量")
        sub2.fft_data = fshift
        sub3 = self.show_image(disp3, f"Step3: 动态范围增强")
        sub3.fft_data = fshift
        
    def ifft2_current(self):
        sub = self.active_sub()
        if not sub:
            return
        fft_data = getattr(sub, "fft_data", None)
        if fft_data is None:
            QMessageBox.warning(self, "FFT反变换", "请先进行FFT操作")
            return
        
        img_reconstructed = ImageProcessor.ifft2(sub.img, fshift=fft_data)
        sub.update(img_reconstructed)
        sub.setWindowTitle(f"FFT反变换 - {sub.windowTitle()}")

    def hist_eq_current(self):
        sub = self.active_sub()
        if not sub:
            return
        img1 = ImageProcessor.hist_eq(sub.img)
        sub1 = self.show_image(img1, f"第一次直方图均衡化")
        
        img2 = ImageProcessor.hist_eq(img1)
        sub2 = self.show_image(img2, f"第二次直方图均衡化")
        
        QMessageBox.information(self, "提示", f"比较两次直方图均衡化的结果是否一致：{np.array_equal(img1, img2)}")

    def homomorphic_current(self):
        sub = self.active_sub()
        if not sub:
            return
        result = ImageProcessor.homomorphic(sub.img)
        self.show_image(result, f"同态滤波 - {sub.windowTitle()}")

    def exp_current(self):
        sub = self.active_sub()
        if not sub:
            return
        gamma, ok = QInputDialog.getDouble(self, "指数增强", "gamma (0.1~3.0)", 1.2, 0.1, 3.0, 2)
        if not ok:
            return
        result = ImageProcessor.exp_transform(sub.img, gamma)
        self.show_image(result, f"指数增强 γ={gamma:.2f}")

    def laplace_current(self):
        sub = self.active_sub()
        if not sub:
            return
        result = ImageProcessor.laplace_sharpen(sub.img)
        self.show_image(result, f"Laplace 锐化 - {sub.windowTitle()}")
        
    def edge_roberts_current(self):
        sub = self.active_sub()
        if not sub:
            return
        result = ImageProcessor.edge_roberts(sub.img)
        self.show_image(result, f"Roberts 边缘检测 - {sub.windowTitle()}")
        
    def edge_prewitt_current(self):
        sub = self.active_sub()
        if not sub:
            return
        result = ImageProcessor.edge_prewitt(sub.img)
        self.show_image(result, f"Prewitt 边缘检测 - {sub.windowTitle()}")
        
    def edge_sobel_current(self):
        sub = self.active_sub()
        if not sub:
            return
        result = ImageProcessor.edge_sobel(sub.img)
        self.show_image(result, f"Sobel 边缘检测 - {sub.windowTitle()}")
        
    def edge_laplace_current(self):
        sub = self.active_sub()
        if not sub:
            return
        result = ImageProcessor.edge_laplace(sub.img)
        self.show_image(result, f"Laplacian 边缘检测 - {sub.windowTitle()}")

    def fd_current(self):
        sub = self.active_sub()
        if not sub:
            return
        items = ImageProcessor.fourier_reconstruct(sub.img, N=100, Ms=(62, 32, 8, 2))
        if not items:
            QMessageBox.warning(self, "傅里叶描述", "未找到闭合边界")
            return

        all_pts = np.vstack([c for _, c in items])
        min_xy = all_pts.min(axis=0) - 10
        max_xy = all_pts.max(axis=0) + 10
        W = int(max_xy[0] - min_xy[0])
        H = int(max_xy[1] - min_xy[1])

        for title, pts in items:
            canvas = np.ones((H, W, 3), np.uint8) * 255
            pts_int = np.round(pts - min_xy).astype(np.int32)
            cv2.polylines(canvas, [pts_int], isClosed=True, color=(0, 0, 0), thickness=1)
            self.show_image(canvas, title)

    def dilate_once_current(self):
        sub = self.active_sub()
        if not sub:
            return
        ImageProcessor.dilate_once_inplace(sub.img)
        sub.update(sub.img)

    def dilate_repeated_current(self):
        sub = self.active_sub()
        if not sub:
            return
        dilated_img = ImageProcessor.dilate_repeated(sub.img, iterations=500)
        self.show_image(dilated_img, f"{sub.windowTitle()} - 反复膨胀")

    def erode_once_current(self):
        sub = self.active_sub()
        if not sub:
            return
        ImageProcessor.erode_once_inplace(sub.img)
        sub.update(sub.img)

    def erode_repeated_current(self):
        sub = self.active_sub()
        if not sub:
            return
        eroded_img = ImageProcessor.erode_repeated(sub.img, iterations=500)
        self.show_image(eroded_img, f"{sub.windowTitle()} - 反复腐蚀")

    def sift_match_current(self):
        sub = self.active_sub()
        if not sub:
            return
        img1 = sub.img

        img2, fname2 = self.get_image()
        if img2 is None:
            return
        
        self.show_image(img2, fname2)

        result = ImageProcessor.sift_match(img1, img2, max_matches=80, ratio=0.75)
        title = f"SIFT 匹配：{sub.windowTitle()} ↔ {fname2}"
        self.show_image(result, title,2, 1)

    def highlight_sub(self,active_sub):
        for sub in self.mdi.subWindowList():
            sub.setStyleSheet("")
        if active_sub:
            active_sub.setStyleSheet("""
    border: 3px solid #0078D7; /* 使用现代蓝色 */
    border-radius: 6px;       /* 添加圆角 */
    background-color: rgb(240, 240, 240); /* 添加半透明背景 */
""")



def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("icon/favicon.ico")))
    win = MainWindow()
    win.show()
    
    # win.show_image(win.get_image(r"image\1.bmp")[0],r"1.bmp")
    # win.show_image(win.get_image(r"image\3.jpg")[0],r"3.jpg")
    # win.show_image(win.get_image(r"image\black_image_2.bmp")[0],r"black_image_2.bmp")
    # win.show_image(win.get_image(r"image\black_image_4.bmp")[0],r"black_image_4.bmp")
    # win.show_image(win.get_image(r"image\black_image_10.bmp")[0],r"black_image_10.bmp")
    # win.show_image(win.get_image(r"image\5.png")[0],r"5.png")
    
    win.show_image(win.get_image(resource_path(r"image\3.jpg"))[0], r"3.jpg")
    win.show_image(win.get_image(resource_path(r"image\black_image_10.bmp"))[0], r"black_image_10.bmp")
    win.show_image(win.get_image(resource_path(r"image\5.png"))[0], r"5.png")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
