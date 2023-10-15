from imageManipulationTools import iManipulate as iMan
import numpy as np

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Define default parameters
        width = 256
        height = 256
        defaultImageArray = np.zeros((width,height,3), np.uint8)

        inputImage  = self.convert2qPixmap(defaultImageArray, (height, width))
        outputImage = self.convert2qPixmap(defaultImageArray, (height, width))

        tx = 0
        ty = 0
        scale = 1
        angle = 0

        # Set main window's properties

        # widgets = [
        #     QDial,
        #     QDoubleSpinBox,
        #     QLabel,
        #     QProgressBar,
        #     QPushButton,
        #     QRadioButton,
        #     QSlider,
        # ]


        widget = QWidget()

        inputImgFigTitle = QLabel("Input Image")
        inputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inputImgFig = QLabel()
        inputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inputImgFig.setPixmap(inputImage)
        
        outputImgFigTitle = QLabel("Output Image")
        outputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outputImgFig = QLabel()
        outputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outputImgFig.setPixmap(outputImage)

        # for w in widgets:
        #     layout.addWidget(w())
        layout = QVBoxLayout()
        layout.addWidget(inputImgFigTitle)
        layout.addStretch()
        layout.addWidget(inputImgFig)
        layout.addStretch()
        layout.addWidget(outputImgFigTitle)
        layout.addStretch()
        layout.addWidget(outputImgFig)

        # Set the layout on the application's window
        widget.setLayout(layout)

        # Set the title for the window
        widget.setWindowTitle("Image Manipulation Tool")

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)


    # TODO: Add a widget that will enable user to browse for image. Take its path and load it. 
    # TODO: Add 4 different widgets that will enable user to transform the image. (Move x, Move y, Scale, Rotate)
    # TODO: Add a button that will enable user apply those transformations and see the output image.


    def loadInputImage(self, path):
        inputImg, size = iMan.loadIm(path)
        inputImg = self.convert2qPixmap(inputImg, size)
        self.updateInputImage(inputImg)


    def updateInputImage(self, qImg):
        self.inputImgFig.setPixmap(qImg)


    def updateOutputImage(self, qImg):
        self.outputImgFig.setPixmap(qImg)


    def convert2qPixmap(self, imgArray, size):
        ConvImg = QImage(imgArray, size[1], size[0], QImage.Format.Format_RGB888) 
        return QPixmap(ConvImg)
    