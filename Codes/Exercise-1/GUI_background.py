from imageManipulationTools import iManipulate as iMan
import numpy as np

from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QStylePainter,
    QColorDialog,
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
        global __width, __height, _defaultImageArray
        __width = 256
        __height = 256
        # self.default_program_mode = "Drawing Mode! Click browse to load an image."
        self.default_program_mode = "Drawing Mode! Click on the input image to palce a point." # When tihs changes, change the manipulation tool to iManCV or iManQT.
        defaultPath = "No image is loaded!"
        _defaultImageArray = np.zeros((__width,__height,3), np.uint8)
        # self.cv2img = _defaultImageArray.copy()
       
        self.inputImage          = self.convert2qPixmap(_defaultImageArray, (__height, __width))
        self.outputImage         = self.convert2qPixmap(_defaultImageArray, (__height, __width))
        self.size = (__height, __width)

        tx = 0
        ty = 0
        scale = 1
        angle = 0
        
        self.circleRadius = 3
        self.circles= []      # [((x0, y0), r0), ((x1, y1), r1), ...]
        self.circles_Tr = []  # [((x0, y0), r0), ((x1, y1), r1), ...]
        self.pointPos = []    # [(x0, y0), (x1, y1), ...]
        # self.pointPos_Tr = [] # [(x0, y0), (x1, y1), ...]

        # Define the coordinate error in the canvas (experimentally determined)
        global __error_x, __error_y
        __error_x = 12
        __error_y = 52


        # Create a container widget
        widget = QWidget()

        # Input Image Labels/Wigdets
        inputImgFigTitle = QLabel("Input Image")
        inputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inputImgFig = QLabel()
        self.inputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inputImgFig.setMaximumSize(QSize(512, 512))
        self.inputImgFig.setPixmap(self.inputImage)
        
        # Output Image Labels/Wigdets
        outputImgFigTitle = QLabel("Output Image")
        outputImgFigTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.outputImgFig = QLabel()
        self.outputImgFig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.outputImgFig.setMaximumSize(QSize(512, 512))
        self.outputImgFig.setPixmap(self.outputImage)

        # Label/Widget for image manipulaton tools
        # Browse = QPushButton("...")
        Browse = QPushButton("Clear!")
        self.programMode = QLabel("{}".format(self.default_program_mode))
        self.programMode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imName = QLabel()
        self.imName.setText("{}".format(defaultPath))
        self.imName.setMinimumWidth(200)
        self.imName.setMaximumWidth(250)
        self.imName.setAlignment(Qt.AlignmentFlag.AlignRight)

        Move_x = QLabel("Move on x axis")
        self.val_x  = QDoubleSpinBox()
        self.val_x.setSuffix(" px")
        self.val_x.setMaximum(1024)
        self.val_x.setMinimum(-1024)
        # print(tx)

        Move_y = QLabel("Move on y axis")
        self.val_y  = QDoubleSpinBox()
        self.val_y.setSuffix(" px")
        self.val_y.setValue(ty)
        self.val_y.setMaximum(1024)
        self.val_y.setMinimum(-1024)
        # print(ty)

        Scale  = QLabel("Scale by factor")
        self.val_s  = QDoubleSpinBox()
        self.val_s.setValue(scale)
        self.val_s.setMaximum(20)
        self.val_s.setMinimum(-20)
        # print(scale)

        Rotate = QLabel("Rotate by angle")
        self.val_r  = QDoubleSpinBox()
        self.val_r.setSuffix(" deg")
        self.val_r.setMaximum(360)
        self.val_r.setMinimum(-360)
        self.val_r.setValue(angle)
        # print(angle)

        # Transform button (Should take all the parameters given and apply transformation)
        Transform = QPushButton("Transform!")


        """ Thing commented here, might be added in the next version. """
        # Browse button functionality
        # Browse.clicked.connect(lambda: self.browseImage()) # Browse for image

        # Browse button functionality (Temporary)
        self.updatePath("Clear the canvas") # Temporary, will be removed in the next version
        Browse.clicked.connect(lambda: self.clearImages()) # Clear the canvas

        # Transformation button functionality (For points)
        Transform.clicked.connect(lambda: self.applyTransformation2points()) # Save value of tx and apply transformation
        print("---------START---------\nInitial parameters are: tx: ",
               tx, " ty: ", ty, " scale: ", scale, " angle: ", angle)
        """"""


        # Place widgets on the window
        layout = QGridLayout()
        layout.addWidget(inputImgFigTitle,   0, 0)
        layout.addWidget(self.inputImgFig,   1, 0, 8, 1)

        layout.addWidget(outputImgFigTitle,  0, 1)
        layout.addWidget(self.outputImgFig,  1, 1, 8, 1)

        layout.addWidget(Browse,             0, 2, 1, 1)
        layout.addWidget(self.imName,        0, 3, 1, 3)
        layout.addWidget(self.programMode,   1, 2, 1, 4)
        layout.addWidget(Move_x,             3, 2, 1, 2)
        layout.addWidget(self.val_x,         4, 2, 1, 2)
        layout.addWidget(Move_y,             3, 4, 1, 2)
        layout.addWidget(self.val_y,         4, 4, 1, 2)
        layout.addWidget(Scale,              5, 2, 1, 2)
        layout.addWidget(self.val_s,         6, 2, 1, 2)
        layout.addWidget(Rotate,             5, 4, 1, 2)
        layout.addWidget(self.val_r,         6, 4, 1, 2)

        layout.addWidget(Transform,          8, 2, 1, 4)

        # Set the layout on the application's window
        widget.setLayout(layout)


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


    """ Thing here, might be added in the next version. """
    # # Import image from OpenCV (3rd party)
    # def loadInputImageFromCV(self, path:str) -> None:
    #     self.cv2img, self.size = iMan.loadIm(path)
    #     self.inputImage = self.convert2qPixmap(self.cv2img, self.size)
    #     self.updateInputImageFig(self.inputImage)
    """"""

    def loadInputImage(self, path:str) -> None:
        self.inputImage = QPixmap(path)
        self.updateInputImageFig(self.inputImage)


    def updatePath(self, path):
        self.imName.setText("{}".format(path))
        # self.programMode.setText("{}".format("Image Transformation Mode!"))


    def updateInputImage(self, qImg):
        self.inputImage = qImg


    def updateOutputImage(self, qImg):
        self.outputImage = qImg


    def updateInputImageFig(self, qImg):
        self.inputImgFig.setPixmap(qImg)


    def updateOutputImageFig(self, qImg):
        self.outputImgFig.setPixmap(qImg)


    def clearInputImageFig(self):
        self.inputImgFig.clear()
        # Clear point position memory
        self.circles= []      # [((x0, y0), r0), ((x1, y1), r1), ...]
        self.pointPos = []    # [(x0, y0), (x1, y1), ...]
        # Create a fresh image and update the input image
        _defaultInputImage  = self.convert2qPixmap(_defaultImageArray, (__height, __width))
        self.updateInputImage(_defaultInputImage)
        self.updateInputImageFig(self.inputImage)


    def clearOutputImageFig(self):
        self.outputImgFig.clear()
        # Clear point position memory
        self.circles_Tr = []  # [((x0, y0), r0), ((x1, y1), r1), ...]
        # self.pointPos_Tr = [] # [(x0, y0), (x1, y1), ...]
        # Create a fresh image and update the output image
        _defaultOutputImage = self.convert2qPixmap(_defaultImageArray, (__height, __width))
        self.updateOutputImage(_defaultOutputImage)
        self.updateOutputImageFig(self.outputImage)


    def clearImages(self): # Clear the canvas - Works only once or twice, fix it!

        # Clear the transformation parameters
        self.resetParameters()
        # Clear image modifications
        self.clearInputImageFig()
        self.clearOutputImageFig()

        print("---------CLEARED---------")


    def updateParameters(self):
        kwarg = {

            "tx"      : self.val_x.value(),
            "ty"      : self.val_y.value(),
            "scale_x" : self.val_s.value(),
            "scale_y" : self.val_s.value(),
            "angle"   : self.val_r.value(),

        }

        """ Thing here, might be added in the next version. """
        #tempImg = self.connectFunc[function](self.cv2img, self.size, **kwarg)
        # self.outputImage = self.convert2qPixmap(tempImg, self.size)
        # self.updateOutputImageFig(self.outputImage)
        # self.resetParameters() # Reset the input parameters after applying the transformation
        """"""
        print("Updated parameters are: ", kwarg)
        return kwarg


    def resetParameters(self):
        self.val_x.setValue(0)
        self.val_y.setValue(0)
        self.val_s.setValue(1)
        self.val_r.setValue(0)


    def applyTransformation2points(self):
        # self.clearOutputImageFig() # Clear the output image - Does not work as intended, fix it!
        param =  self.updateParameters()
        self.clearOutputImageFig()
        color_Tr = (0, 255, 0, 127)
        for point in self.pointPos:
            tempPos = tuple(iMan.manipulateP(point, **param))
            tempPos_x = tempPos[0]
            tempPos_y = tempPos[1]
            # If the mouse click is inside the input image borders
            if (tempPos_x > __error_x and tempPos_x < __width + __error_x) and (tempPos_y > __error_y and tempPos_y < __height + __error_y):
                self.circles_Tr.append((tempPos, self.circleRadius))
        
        print("Original points: ",    self.circles)
        print("Transformed points: ", self.circles_Tr)
        self.draw_circles(self.outputImage, self.circles_Tr, color_Tr, output_Image=True)
        self.updateOutputImageFig(self.outputImage)
        # self.resetParameters() # Reset the input parameters after applying the transformation


    def convert2qPixmap(self, imgArray, size):
        ConvImg = QImage(imgArray, size[1], size[0], QImage.Format.Format_RGB888) 
        return QPixmap(ConvImg)


    def draw_circles(self, img, circle_pos=None, color=(255, 0, 0, 127), output_Image = False):
        painter = QPainter(img)
        # Error in the coordinates of placed circle has been determined experimentally
        err_x = 16
        err_y = 54
        for pos, radius in circle_pos:
            x, y = pos
            painter.setBrush(QColor(*color))  # Set color
            painter.drawEllipse(x - err_x, y - err_y, 2 * radius, 2 * radius)
        if output_Image:
            self.updateOutputImage(img)
        else:
            self.updateInputImage(img)


    # Define events

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            tempPos_x = int(event.position().x())
            tempPos_y = int(event.position().y())
            # print("Clicked at the position: ", tempPos_x, tempPos_y)
            # If the mouse click is inside the input image borders
            if (tempPos_x > __error_x and tempPos_x < __width + __error_x) and (tempPos_y > __error_y and tempPos_y < __height + __error_y):
                self.pointPos.append([tempPos_x, tempPos_y])
                # Add a circle to the list
                self.circles.append(((tempPos_x, tempPos_y), self.circleRadius))  # (position, radius)
                # Add the cricle in the image
                self.draw_circles(self.inputImage, self.circles, output_Image=False)
                # Redraw the image with the new circle
                self.updateInputImageFig(self.inputImage)
         

# [On Hold] TODO: Fix the bug that causes groot to be warped, and others to change colors.
# TODO: Add the functionality to put circles on the image and transform them as well.
    