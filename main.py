from logging import raiseExceptions
from turtle import color
from PIL import Image, ImageDraw, ImageFilter
import PyQt5
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QComboBox, QHBoxLayout, QPushButton, QGridLayout, QLabel, QLineEdit, QFileDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from yaml import load
from colorifer import colorifer
import sys
from os import remove, getcwd
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


blur_list = ['Gaussian', 'BoxBlur']

img_status = ['Transformed', 'Black-white', 'Labeled']

class MplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=1, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class app_window(QWidget):
    def __init__(self):
        super().__init__()
        self.height = 500
        self.width = 500
        self.create_none_image()
        self.base_image = Image.open('none-image.jpg').convert('RGB')
        self.transformed_image = self.base_image.copy()
        self.bw_image = self.base_image.copy()
        self.labeled_image = self.base_image.copy()
        self.none_image = self.base_image.copy()
        self.pie_percent = [1]
        self.pie_colors = np.array((255, 255, 255))
        self.filename = None
        
        
        self.clrfr = colorifer()
        
        self.init_gui()
        
    def init_gui(self):
        self.resize(1200, 600)
        self.setWindowTitle('Image colorifer')
        
        app_layout = QHBoxLayout()
        
        load_groupbox = QGroupBox('Import image')
        load_grid = QGridLayout()
        
        load_btn = QPushButton('Load Image')
        load_btn.resize(load_btn.sizeHint())
        load_btn.clicked.connect(self.clicked_load_image)
        load_grid.addWidget(load_btn, 0, 0)
        
        self.base_image_label = QLabel()
        base_image_pixmap = QPixmap('none-image.jpg').scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.base_image_label.setPixmap(base_image_pixmap)
        self.base_image_label.setFixedSize(self.width, self.height)
        load_grid.addWidget(self.base_image_label, 1, 0)
        
        self.reset_img_btn = QPushButton('Reset image')
        self.reset_img_btn.resize(self.reset_img_btn.sizeHint())
        self.reset_img_btn.clicked.connect(self.clicked_reset_image)
        load_grid.addWidget(self.reset_img_btn)
        
        load_groupbox.setLayout(load_grid)
               
        transform_groupbox = QGroupBox('Transformation parameters')
        self.transform_grid = QGridLayout()
        self.transform_grid.addWidget(QLabel('N colors: '), 0, 0)
        self.n_colors_le = QLineEdit()
        self.n_colors_le.setText('8')
        self.n_colors_le.resize(self.n_colors_le.sizeHint())
        self.transform_grid.addWidget(self.n_colors_le, 0, 1)
        
        self.transform_grid.addWidget(QLabel('Scale: '), 1, 0)
        self.scale_le = QLineEdit('1.0')
        self.scale_le.resize(self.scale_le.sizeHint())
        self.transform_grid.addWidget(self.scale_le, 1, 1)
        
        self.transform_grid.addWidget(QLabel('Blur filter: '), 2, 0)
        self.blur_combo = QComboBox()
        self.blur_combo.addItems(blur_list)
        self.blur_combo.resize(self.blur_combo.sizeHint())
        self.transform_grid.addWidget(self.blur_combo, 2, 1)
        self.transform_grid.addWidget(QLabel('Parameter: '), 2, 2)
        self.blur_param_le = QLineEdit('3')
        self.blur_param_le.resize(self.blur_param_le.sizeHint())
        self.transform_grid.addWidget(self.blur_param_le, 2, 3)
        
        self.transform_grid.addWidget(QLabel('Noise filter size: '), 3, 0)
        self.noise_filter_size_le = QLineEdit('3')
        self.noise_filter_size_le.resize(self.noise_filter_size_le.sizeHint())
        self.transform_grid.addWidget(self.noise_filter_size_le, 3, 1)
        self.transform_grid.addWidget(QLabel('Noise filters number: '), 3, 2)
        self.noise_filter_number_le = QLineEdit('1')
        self.noise_filter_number_le.resize(self.noise_filter_number_le.sizeHint())
        self.transform_grid.addWidget(self.noise_filter_number_le, 3, 3)
        
        self.transform_btn = QPushButton('Transform')
        self.transform_btn.resize(self.transform_btn.sizeHint())
        self.transform_btn.clicked.connect(self.clicked_transform)
        self.transform_grid.addWidget(self.transform_btn, 4, 0, 1, 4)
        
        self.transform_reset_btn = QPushButton('Reset parameters')
        self.transform_reset_btn.resize(self.transform_reset_btn.sizeHint())
        self.transform_reset_btn.clicked.connect(self.clicked_transform_reset)
        self.transform_grid.addWidget(self.transform_reset_btn, 5, 0, 1, 4)
        
        self.pie_chart = MplCanvas(self, width=1, height=1, dpi=100)
        self.pie_chart.axes.pie(self.pie_percent, colors=[self.pie_colors/255])
        self.transform_grid.addWidget(self.pie_chart, 6, 0, 1, 4)
        
        transform_groupbox.setLayout(self.transform_grid)
        
        result_groupbox = QGroupBox()
        result_grid = QGridLayout()
        
        self.result_combo = QComboBox()
        self.result_combo.addItems(img_status)
        self.result_combo.resize(self.result_combo.sizeHint())
        self.result_combo.currentIndexChanged.connect(self.result_combo_changed)
        result_grid.addWidget(self.result_combo, 0, 0)
        result_groupbox.setLayout(result_grid)
        
        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self.clicked_save)
        self.save_btn.resize(self.save_btn.sizeHint())
        result_grid.addWidget(self.save_btn, 0, 1)
        
        self.save_all_btn = QPushButton('Save all')
        self.save_all_btn.clicked.connect(self.clicked_save_all)
        self.save_all_btn.resize(self.save_all_btn.sizeHint())
        result_grid.addWidget(self.save_all_btn, 0, 2)
        
        self.result_image_label = QLabel()
        result_image_pixmap = QPixmap('none-image.jpg').scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.result_image_label.setPixmap(result_image_pixmap)
        self.result_image_label.setFixedSize(self.width, self.height)
        result_grid.addWidget(self.result_image_label, 1, 0, 1, 3)
        
        self.reset_results_btn = QPushButton('Reset results')
        self.reset_results_btn.clicked.connect(self.clicked_reset_results)
        self.reset_results_btn.resize(self.reset_results_btn.sizeHint())
        result_grid.addWidget(self.reset_results_btn, 2, 0, 1, 3)
      
        
        app_layout.addWidget(load_groupbox)
        app_layout.addWidget(transform_groupbox)
        app_layout.addWidget(result_groupbox)
        self.setLayout(app_layout)
        
        remove('none-image.jpg')
        
    def create_none_image(self):
        img = Image.new('RGB', (self.width, self.height), color = 'white')
        draw_text = ImageDraw.Draw(img)
        draw_text.text(
                    (int(self.width/2-50), int(self.height/2)),
                    'Image is not loaded',
                    fill=('#1C0606')
                    )
        img.save('./none-image.jpg')
    
    def clicked_load_image(self):
        dlg = QFileDialog()
        
        if dlg.exec_():
          self.filename = dlg.selectedFiles()[0]
          try:
            self.base_image = Image.open(self.filename).convert('RGB')
            base_image_pixmap = QPixmap(self.filename).scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
            self.base_image_label.setPixmap(base_image_pixmap)
          except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Unable to load image')
            msg.setWindowTitle("Error")
            msg.exec_()
        
    
    def clicked_transform(self):
        params = 0
        try:
            n_colors = int(self.n_colors_le.text())
            params += 1
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Wrong number of colors')
            msg.setWindowTitle("Error")
            msg.exec_()
            
        try:
            scale = float(self.scale_le.text())
            if scale > 0:
                params += 1
            else:
                raiseExceptions()
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Wrong scale')
            msg.setWindowTitle("Error")
            msg.exec_()
            
        try:
            blur_param = int(self.blur_param_le.text())
            if blur_param > 0:
                params += 1
            else:
                raiseExceptions()
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Wrong blurring parameter')
            msg.setWindowTitle("Error")
            msg.exec_()
            
        try:
            noise_filter_size = int(self.noise_filter_size_le.text())
            if noise_filter_size > 0:
                params += 1
            else:
                raiseExceptions()
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Wrong noise filter size')
            msg.setWindowTitle("Error")
            msg.exec_()
            
        try:
            noise_filter_number = int(self.noise_filter_number_le.text())
            if noise_filter_number > 0:
                params += 1
            else:
                raiseExceptions()
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Wrong noise filter number')
            msg.setWindowTitle("Error")
            msg.exec_()
            
        if params == 5:
            if self.blur_combo.currentIndex() == 0:
                blurrer = ImageFilter.GaussianBlur(blur_param)
            else:
                blurrer = ImageFilter.BoxBlur(blur_param)
            
            if self.filename is not None:
                self.clrfr = colorifer(n_colors, scale, blurrer, noise_filter_size, noise_filter_number)
                self.clrfr.fit(self.filename)
                
                self.transform_grid.removeWidget(self.pie_chart)
                self.pie_chart = MplCanvas(self, width=1, height=1, dpi=100)
                self.pie_chart.axes.pie(self.clrfr.pie_percent,colors=np.array(self.clrfr.pie_centroid/255), labels=np.arange(len(self.clrfr.pie_centroid)))
                self.transform_grid.addWidget(self.pie_chart, 6, 0, 1, 4)
                                
                new_im_data = self.clrfr.transform()
                self.transformed_image = Image.fromarray(np.uint8(new_im_data)).convert('RGB')
                bw_data = self.clrfr.transform_bw()
                self.bw_image = Image.fromarray(np.uint8(bw_data)).convert('RGB')
                labeled_data = self.clrfr.add_labels()
                self.labeled_image = Image.fromarray(np.uint8(labeled_data)).convert('RGB')
                
                temp_images = [self.transformed_image, self.bw_image, self.labeled_image]
                new_image_mode = self.result_combo.currentIndex()
                temp_images[new_image_mode].save('temporary_image.jpg')
                
                result_image_pixmap = QPixmap('temporary_image.jpg').scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
                self.result_image_label.setPixmap(result_image_pixmap)
                remove('temporary_image.jpg')
                
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Image is not chosen')
                msg.setWindowTitle("Error")
                msg.exec_()
    
    def clicked_transform_reset(self):
        self.n_colors_le.setText('8')
        self.scale_le.setText('1.0')
        self.blur_combo.setCurrentIndex(0)
        self.blur_param_le.setText('3')
        self.noise_filter_size_le.setText('3')
        self.noise_filter_number_le.setText('1')
    
    def clicked_save(self):
        try:
            name = QFileDialog.getSaveFileName(self, 'Save File')
            if self.result_combo.currentIndex() == 0:
                self.transformed_image.save(name[0] + '.jpg')
            elif self.result_combo.currentIndex() == 1:
                self.bw_image.save(name[0] + '.jpg')
            else:
                self.labeled_image.save(name[0] + '.jpg')
        except Exception as e:
            print(e)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Unable to save here')
            msg.setWindowTitle("Error")
            msg.exec_()        
            
    
    def clicked_save_all(self):
        try:
            name = QFileDialog.getSaveFileName(self, 'Save File')
            self.transformed_image.save(name[0] +'_transformed_.jpg')
            self.bw_image.save(name[0] +'_bw.jpg')
            self.labeled_image.save(name[0] +'_labeled.jpg')
        except Exception as e:
            print(e)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Unable to save here')
            msg.setWindowTitle("Error")
            msg.exec_()   
        
        self.transformed_image.save('img1.jpg')
        self.bw_image.save('img2.jpg')
        self.labeled_image.save('img3.jpg')
    
    def clicked_reset_image(self):
        self.create_none_image()
        self.base_image = Image.open('none-image.jpg').convert('RGB')
        base_image_pixmap = QPixmap('none-image.jpg').scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.base_image_label.setPixmap(base_image_pixmap)
        remove('none-image.jpg')
    
    def clicked_reset_results(self):
        self.create_none_image()
        self.result_image = Image.open('none-image.jpg').convert('RGB')
        result_image_pixmap = QPixmap('none-image.jpg').scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.result_image_label.setPixmap(result_image_pixmap)
        remove('none-image.jpg')        
       
    
    def result_combo_changed(self, value):
        temp_images = [self.transformed_image, self.bw_image, self.labeled_image]
        new_image_mode = self.result_combo.currentIndex()
        temp_images[new_image_mode].save('temporary_image.jpg')
                
        result_image_pixmap = QPixmap('temporary_image.jpg').scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.result_image_label.setPixmap(result_image_pixmap)
        remove('temporary_image.jpg')
          
    
app = QApplication(sys.argv)
window = app_window()
window.show()
sys.exit(app.exec_())
