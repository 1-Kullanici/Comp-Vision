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
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
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
        
        path = "No image is loaded!"

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

        # Input Image Labels/Wigdets
        inputImgFigTitle = QLabel("Input Image")
        inputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inputImgFig = QLabel()
        inputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inputImgFig.setPixmap(inputImage)
        
        # Output Image Labels/Wigdets
        outputImgFigTitle = QLabel("Output Image")
        outputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outputImgFig = QLabel()
        outputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outputImgFig.setPixmap(outputImage)

        # Label/Widget for image manipulaton tools
        Browse = QPushButton("...")
        imName = QLabel()
        imName.setText("{}".format(path))
        imName.setAlignment(Qt.AlignmentFlag.AlignRight)

        Move_x = QPushButton("Move on x axis")
        val_x  = QDoubleSpinBox()
        val_x.setSuffix(" px")
        val_x.setMaximum(255)
        val_x.setMinimum(-255)
        val_x.setValue(tx)
        print(tx)

        Move_y = QPushButton("Move on y axis")
        val_y  = QDoubleSpinBox()
        val_y.setSuffix(" px")
        val_y.setValue(ty)
        val_y.setMaximum(255)
        val_y.setMinimum(-255)
        print(ty)

        Scale  = QPushButton("Scale by factor")
        val_s  = QDoubleSpinBox()
        val_s.setValue(scale)
        val_s.setMaximum(255)
        val_s.setMinimum(-255)
        print(scale)

        Rotate = QPushButton("Rotate by angle")
        val_r  = QDoubleSpinBox()
        val_r.setSuffix(" deg")
        val_r.setMaximum(360)
        val_r.setMinimum(-360)
        val_r.setValue(angle)
        print(angle)

        # Place widgets on the window
        layout = QGridLayout()
        layout.addWidget(inputImgFigTitle, 0, 0)
        layout.addWidget(inputImgFig, 1, 0, 8, 1)

        layout.addWidget(outputImgFigTitle, 0, 1)
        layout.addWidget(outputImgFig, 1, 1, 8, 1)

        layout.addWidget(Browse, 0, 2, 1, 1)
        layout.addWidget(imName, 0, 3, 1, 3)
        layout.addWidget(Move_x, 1, 2, 1, 4)
        layout.addWidget(val_x,  2, 2, 1, 4)
        layout.addWidget(Move_y, 3, 2, 1, 4)
        layout.addWidget(val_y,  4, 2, 1, 4)
        layout.addWidget(Scale,  5, 2, 1, 4)
        layout.addWidget(val_s,  6, 2, 1, 4)
        layout.addWidget(Rotate, 7, 2, 1, 4)
        layout.addWidget(val_r,  8, 2, 1, 4)


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
    