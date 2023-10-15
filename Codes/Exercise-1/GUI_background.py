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
    QFileDialog,
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Define default parameters
        width = 256
        height = 256
        defaultPath = "No image is loaded!"
        defaultImageArray = np.zeros((width,height,3), np.uint8)
        self.cv2img = defaultImageArray.copy()

        self.inputImage  = self.convert2qPixmap(defaultImageArray, (height, width))
        self.outputImage = self.convert2qPixmap(defaultImageArray, (height, width))
        self.size = (height, width)

        tx = 0
        ty = 0
        scale = 1
        angle = 0
        

        # Create a container widget
        widget = QWidget()

        # Input Image Labels/Wigdets
        inputImgFigTitle = QLabel("Input Image")
        inputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inputImgFig = QLabel()
        self.inputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inputImgFig.setPixmap(self.inputImage)
        
        # Output Image Labels/Wigdets
        outputImgFigTitle = QLabel("Output Image")
        outputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.outputImgFig = QLabel()
        self.outputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.outputImgFig.setPixmap(self.outputImage)

        # Label/Widget for image manipulaton tools
        Browse = QPushButton("...")
        self.imName = QLabel()
        self.imName.setText("{}".format(defaultPath))
        self.imName.setMinimumWidth(200)
        self.imName.setMaximumWidth(250)
        self.imName.setAlignment(Qt.AlignmentFlag.AlignRight)

        Move_x = QPushButton("Move on x axis")
        self.val_x  = QDoubleSpinBox()
        self.val_x.setSuffix(" px")
        self.val_x.setMaximum(255)
        self.val_x.setMinimum(-255)
        # print(tx)

        Move_y = QPushButton("Move on y axis")
        self.val_y  = QDoubleSpinBox()
        self.val_y.setSuffix(" px")
        self.val_y.setValue(ty)
        self.val_y.setMaximum(255)
        self.val_y.setMinimum(-255)
        # print(ty)

        Scale  = QPushButton("Scale by factor")
        self.val_s  = QDoubleSpinBox()
        self.val_s.setValue(scale)
        self.val_s.setMaximum(255)
        self.val_s.setMinimum(-255)
        # print(scale)

        Rotate = QPushButton("Rotate by angle")
        self.val_r  = QDoubleSpinBox()
        self.val_r.setSuffix(" deg")
        self.val_r.setMaximum(360)
        self.val_r.setMinimum(-360)
        self.val_r.setValue(angle)
        # print(angle)


        # Map widgets to functions
        self.connectFunc = {

            Move_x  : iMan.translateIm,
            Move_y  : iMan.translateIm,
            Scale   : iMan.scaleIm,
            Rotate  : iMan.rotateIm,
            
        }


        # Browse button functionality
        Browse.clicked.connect(lambda: self.browseImage()) # Browse for image


        # Connect widgets to functions [FIXED] -Use lamda: when running a void function inside another function.-|--#TODO: Does not work as intended, fix it!#--
        Move_x.clicked.connect(lambda: self.updateParameters(Move_x)) # Save value of tx and apply transformation
        Move_y.clicked.connect(lambda: self.updateParameters(Move_y)) # Save value of ty and apply transformation
        Scale.clicked.connect(lambda:  self.updateParameters(Scale))  # Save value of scale and apply transformation
        Rotate.clicked.connect(lambda: self.updateParameters(Rotate)) # Save value of angle and apply transformation
        print("---------START---------\nInitial parameters are: tx: ",
               tx, " ty: ", ty, " scale: ", scale, " angle: ", angle)


        # Place widgets on the window
        layout = QGridLayout()
        layout.addWidget(inputImgFigTitle, 0, 0)
        layout.addWidget(self.inputImgFig, 1, 0, 8, 1)

        layout.addWidget(outputImgFigTitle, 0, 1)
        layout.addWidget(self.outputImgFig, 1, 1, 8, 1)

        layout.addWidget(Browse, 0, 2, 1, 1)
        layout.addWidget(self.imName, 0, 3, 1, 3)
        layout.addWidget(Move_x, 1, 2, 1, 4)
        layout.addWidget(self.val_x,  2, 2, 1, 4)
        layout.addWidget(Move_y, 3, 2, 1, 4)
        layout.addWidget(self.val_y,  4, 2, 1, 4)
        layout.addWidget(Scale,  5, 2, 1, 4)
        layout.addWidget(self.val_s,  6, 2, 1, 4)
        layout.addWidget(Rotate, 7, 2, 1, 4)
        layout.addWidget(self.val_r,  8, 2, 1, 4)


        # Set the layout on the application's window
        widget.setLayout(layout)

        # Set the title for the window
        widget.setWindowTitle("Image Manipulation Tool")

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)


    # [DONE] TODO: Add a widget that will enable user to browse for image. Take its path and load it. 
    # [DONE] TODO: Add 4 different widgets that will enable user to transform the image. (Move x, Move y, Scale, Rotate)
    # [DONE] TODO: Add a button that will enable user apply those transformations and see the output image.


    def browseImage(self):
        self.fileName, _ = QFileDialog.getOpenFileName(None, 'Open a File', '', 'Image files (*.jpg *.png *.jpeg *.bmp *.tif *.tiff)')
        if self.fileName is not None:
            self.updatePath(self.fileName)
            self.loadInputImage(self.fileName)


    def loadInputImage(self, path):
        self.cv2img, self.size = iMan.loadIm(path)
        self.inputImg = self.convert2qPixmap(self.cv2img, self.size)
        self.updateInputImage(self.inputImg)


    def updatePath(self, path):
        self.imName.setText("{}".format(path))


    def updateInputImage(self, qImg):
        self.inputImgFig.setPixmap(qImg)


    def updateOutputImage(self, qImg):
        self.outputImgFig.setPixmap(qImg)


    def updateParameters(self, function: object):
        kwarg = {

            "tx"      : self.val_x.value(),
            "ty"      : self.val_y.value(),
            "scale_x" : self.val_s.value(),
            "scale_y" : self.val_s.value(),
            "angle"   : self.val_r.value(),

        }

        # Test if the variables are passed correctly
        print("Updated parameters:")
        print(kwarg)

        tempImg = self.connectFunc[function](self.cv2img, self.size, **kwarg)
        self.outputImage = self.convert2qPixmap(tempImg, self.size)
        self.updateOutputImage(self.outputImage)
        self.resetParameters() # Reset the input parameters after applying the transformation
        

    def resetParameters(self):
        self.val_x.setValue(0)
        self.val_y.setValue(0)
        self.val_s.setValue(1)
        self.val_r.setValue(0)


    def convert2qPixmap(self, imgArray, size):
        ConvImg = QImage(imgArray, size[1], size[0], QImage.Format.Format_RGB888) 
        return QPixmap(ConvImg)
    

# TODO: Fix the bug that causes groot to be warped, and others to change colors.
# TODO: Add the functionality to put circles on the image and transform them as well.
    