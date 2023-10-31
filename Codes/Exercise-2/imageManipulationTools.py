import argparse
import cv2 as cv
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


class iManipulate:
    """
        This class manipulates images and points using functions I wrote.
    """
    def __init__(self):
        pass

    def manipulateP(point:tuple, tx:float=0, ty:float=0, scale_x:float=1, scale_y:float=1, angle:float=0) -> list: 
        """
            manipulateP(point, tx=0, ty=0, scale_x=1, scale_y=1, angle=0)
            
        Takes point coordinates and transformation parameters as input and returns the coordinate of the transformed point as list.
        
        Parameters:
        --------------------------------
        point: tuple
        tx: float
        ty: float
        scale_x: float
        scale_y: float
        angle: float (in degrees)
        """
        angle_in_Radians = angle*np.pi/180
        cosPart = np.cos(angle_in_Radians)
        sinPart = np.sin(angle_in_Radians)
        point_ext = np.array([point[0], point[1], 1])
        # Translation matrix ([tanslate * scale * rotate])
        tM = np.float32([[scale_x * cosPart, -(scale_y * sinPart), tx * scale_x * cosPart - ty * scale_y * sinPart], 
                         [scale_x * sinPart, scale_y * cosPart,    tx * scale_x * sinPart + ty * scale_y * cosPart],
                         [0,                 0,                    1                                              ]])
        
        new_point_ext = np.matmul(tM, point_ext)
        new_point = np.array([int(new_point_ext[0]), int(new_point_ext[1])])
        return new_point

    def manipulateIm(image:list, tx:float=0, ty:float=0, scale_x:float=1, scale_y:float=1, angle:float=0, channel:int=1) -> list:
        """
            manipulateIm(image, tx=0, ty=0, scale_x=1, scale_y=1, angle=0, channel=1)
            
        Takes image and transformation parameters as input and returns the transformed image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.
        
        Parameters:
        --------------------------------
        image: list
        tx: float
        ty: float
        scale_x: float
        scale_y: float
        angle: float (in degrees)
        channel: int
        """
        new_image = image
        print("---------ImgTransform---------\nAdd functionality to manipulateIm() function.")
        return new_image

    def processIm(img:list, channel:int=1, mode:str='', radius:int=1, gamma:float=1, padding=True) -> list:
        """
            processIm(image, channel=1, mode='', radius=1, customKernel=[1])
            
        Takes image, channel, mode, kernel radius, custom kernel, and padding as input and returns the processed image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.\n
        If no mode is selected, it returns the image as it is.\n
        If padding is enabled, the output image will not clipped.
        
        Available Modes:
        --------------------------------
        - Linear
        -- Prewitt edge detection

        - Non-linear
        -- Gamma correction
        -- DCT (discrete cosine transform)
        
        - Custom
        
        Parameters:
        --------------------------------
        image: list
        channel: int
        mode: str
        radius: int
        customKernel: list
        padding: bool
        """
        # Check mode
        if mode == '':
            return img
        else:
            image = img
            channel = channel
            gamma = gamma
            definedModes = {
                'gamma'   : iProcess.gammaCorrection(img=image, channel=channel, gamma=gamma),
                'prewitt' : iProcess.prewittEdgeDetection(img=image, channel=channel),
                'dct'     : iProcess.dct2d(image=image, height=..., width=..., channel=channel) 
            }
            processedIm = definedModes[mode]
            return processedIm


    # def ...(img:list, channel:int=1) -> list:
    #     pass


class iProcess:
    """
        This class contains various image processing methods.
    """       
    def __init__(self):
        pass
    
    def gammaCorrection(self, img:list, channel:int=1, gamma:float=1) -> list:
        """
            gammaCorrection(image, channel=1, gamma=1)

        Takes image, channel, and gamma parameters as input and returns the gamma corrected image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.

        Parameters:
        --------------------------------
        image: list
        channel: int
        gamma: float
        """
        gammaCorrectedImg = np.array([np.power(img[ch], gamma) for ch in channel])
        return gammaCorrectedImg
    
    def prewittEdgeDetection(self, img:list, channel:int=1) -> list:
        """
            prewittEdgeDetection(image, channel=1)

        Takes image and channel parameters as input and returns the edge detected image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.

        Parameters:
        --------------------------------
        image: list
        channel: int
        """
        Gx = []
        Gy = []
        G  = []
        r = 3
        edge_x = iWrap.kernelWrapper('prewitt_x')
        edge_y = iWrap.kernelWrapper('prewitt_y')
        parcedImg = iWrap.imageParser(img, channel, radius=r, padding=True)
        if channel == 1:
            multiplied_subs = np.einsum('ij,ijkl->ijkl',edge_x,parcedImg)
            Gx.append(np.sum(np.sum(multiplied_subs, axis = -r), axis = -r))
            
        else:
            for ch in range(channel):
                dx = np.einsum('ij,ijkl->ijkl',edge_x,parcedImg[ch])
                Gx.append(np.sum(np.sum(dx, axis = -r), axis = -r))
                dy = np.einsum('ij,ijkl->ijkl',edge_y,parcedImg[ch])
                Gy.append(np.sum(np.sum(dy, axis = -r), axis = -r))
                G.append(np.sqrt(np.power(Gx,2)+np.power(Gy,2)))

        return G

    def dct1d(self, x:list, channel:int=1, array_length:int=-1) -> list:
        """
            dct1d(x, channel=1, array_length=-1)

        Takes a one dimentional array and channel parameters as input and returns the discrete cosine transformed image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.

        Parameters:
        --------------------------------
        x: list
        channel: int
        array_length: int

        Snippet origin: https://github.com/diegolrs/DCT2D-Digital-Image-Processing.git
        """
        if array_length == -1:
            length = len(x)
        else:
            length = array_length

        m_X = [0 for j in range(length)]

        alpha = np.math.pow(2/length, 0.5)

        for k in range(length):
            if k == 0:
                ck = np.math.pow(1/2, 1/2)
            else:
                ck = 1

            _sum = 0

            for n in range(length):
                _temp = (np.math.pi * k) / (2*length)
                _cos = np.math.cos((2*n*_temp) + _temp) 
                _sum = _sum + (_cos * x[n])

            m_X[k] = alpha * ck * _sum

        return m_X
    
    def dct2d(self, image, height, width, channel=1):
        """
            dct2d(image, height, width, channel=1)

        Takes a two dimentional array and channel parameters as input and returns the discrete cosine transformed image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.

        Parameters:
        --------------------------------
        image: list
        height: int
        width: int
        channel: int

        Snippet origin: https://github.com/diegolrs/DCT2D-Digital-Image-Processing.git
        """
        new_image = []
        for ch in channel:
            #applying dct1d row by row
            tempArrRow = np.array(image[ch]).tolist()
            for i in range(height):
                tempArrRow[i] = self.dct1d(tempArrRow[i])
            #applying dct1d column by column
            tempArrCol = np.array(tempArrRow)
            for j in range(width):
                tempArrCol[ :, j] = self.dct1d(tempArrCol[ :, j])

            new_image.append(tempArrCol)
        return new_image


class iWrap:
    """
        This class contains wrappers.
    """

    def color_clamped(self, color):
        """
            color_clamped(color)
            Takes color as input and returns the clamped color as output.

            Snippet origin: https://github.com/diegolrs/DCT2D-Digital-Image-Processing.git
        """
        return max(min(color, 255), 0)

    def imageParser(self, img:list, channel:int=1, radius:int=1, padding=True) -> list:
        """
            imageParser(image, channel=1, radius=1)

        Takes image, channel, and radius parameters as input and returns the image parsed by radius as a list of lists.\n
        If channel is 1 (set by default), it returns a parsed grayscale image. If channel is 3, it returns a parsed color image.

        Parameters:
        --------------------------------
        image: list
        channel: int
        radius: int
        """
        image = np.pad(img, radius, mode='constant') if padding else img
        sub_shape = (radius, radius)
        parsedCh = np.empty(radius)
        for ch in channel:
            temp = tuple(np.subtract(image[ch].shape, sub_shape) + 1) + sub_shape
            temp2 = ast(image[ch], temp, image[ch].strides * 2)
            parsedCh[ch] = list(temp2.reshape((-1,) + sub_shape))
        
        return parsedCh 
    
    def kernelWrapper(type:str='', radius:int=1) -> list:
        """
            kernelWrapper(type, radius=1)
            
        Takes type, radius, and custom kernel as input and returns the desired kernel as a square list.\n
        Type defines the kernel type desired.
        If no type is selected it returns identity matrix.
        
        The pre-defined kernels are:
        --------------------------------
        - Identity (inactive)                                   - ''
        - Prewitt edge detection (neighborhood operation)       - prewitt_x, prewitt_y
        - DCT (discrete cosine transform) (global operation)    - dct
        
        Parameters:
        --------------------------------
        type: str
        radius: int
        """
        identityKernel = np.identity(radius)
        predefinedKernels = {
            'prewitt_x': np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
            'prewitt_y': np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
        }
        if type == '':
            return identityKernel
        else:
            try:
                return predefinedKernels[type]
            except:
                print("Kernel type not found. Returning identity kernel.")
                return identityKernel
            

def main():

    # # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # args = vars(ap.parse_args())

    imPath = "/Users/ift/Desktop/Belgeler ve klasörler/Görseller/Groot_ama_sisko.jpg"
    in_window_name = "inputImage"
    out_window_name = "outputImage"

    inputImg, size = iManipulate.loadIm(imPath)
    print(size)

    # Create seperate windows for input image and output image
    cv.namedWindow(in_window_name)
    cv.namedWindow(out_window_name)

    # Set mouse callback function for input image
    cv.setMouseCallback("{}".format(in_window_name), iManipulate.draw_circle)

    # Show input image and wait for user input
    while 1:
        cv.imshow(in_window_name, inputImg)
        cv.waitKey(1)
        if cv.waitKey(20) & 0xFF == 27:
            break


    # Test output images
    outputImg = iManipulate.translateIm(inputImg, size, 100, -100)
    iManipulate.showIm(out_window_name, outputImg)
    outputImgR = iManipulate.rotateIm(inputImg, size, 45)
    iManipulate.showIm(out_window_name, outputImgR)
    outputImgS = iManipulate.scaleIm(inputImg, size, .5, .5)
    iManipulate.showIm(out_window_name, outputImgS)


if __name__ == "__main__":
    main()


#TODO: Update input image when a marker is placed. Transform the markers to output image. Add these to GUI. 