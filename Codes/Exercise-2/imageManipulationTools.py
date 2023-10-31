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
                'gamma'   : iProcess.gammaCorrection(image, gamma),
                # 'prewitt' : iProcess.prewittEdgeDetection(image), # Takes too long to process
                'dct'     : iProcess.dct2d(image=image) 
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
    
    def gammaCorrection(img:list, gamma:float=1) -> list:
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
        (row, col, channel) = img.shape
        gammaCorrectedImg = np.zeros((row, col, channel), dtype=np.uint8)
        # for r in range(row):
        #     for c in range(col):
        #         for ch in range(channel):
        #             gammaCorrectedImg[r][c][ch] = img[r][c][ch] ** (1/gamma)
        for ch in range(channel):
            gammaCorrectedImg[:,:,ch] = img[:,:,ch] ** (1/gamma) 
            # print(":D")
        # print(gammaCorrectedImg)
        return gammaCorrectedImg
    
    def prewittEdgeDetection(img:list) -> list:
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
        (row, col, channel) = img.shape
        edge_x = iWrap.kernelWrapper('prewitt_x')
        edge_y = iWrap.kernelWrapper('prewitt_y')
        image  = iWrap.cvIm2List(img, r, padding=True)
        parcedImg = iWrap.imageParser(image, row, col, channel, radius=r, padding=False)
        channel = len(parcedImg)
        (row, col, *_) = parcedImg[0].shape
        # print(row, col, channel)
        for ch in range(channel):
            for i in range(row):
                for j in range(col):
                    dx = np.matmul(edge_x,parcedImg[ch][i][j])
                    Gx.append(np.sum(np.sum(dx, axis = -1), axis = -1))
                    dy = np.matmul(edge_y,parcedImg[ch][i][j])
                    Gy.append(np.sum(np.sum(dy, axis = -1), axis = -1))
                    G.append(np.sqrt(np.power(Gx,2)+np.power(Gy,2)))

        return G

    def dct1d(x:list, channel:int=1, array_length:int=-1) -> list:
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
    
    def dct2d(image):
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
        image = iWrap.cvIm2List(image, padding=False)
        channel, *_ = image.shape
        # print(channel)
        height, width = image[0].shape
        # print(height, width)
        new_image = np.empty((channel), dtype=np.ndarray)
        for ch in range(channel):
            #applying dct1d row by row
            tempArrRow = np.array(image[ch]).tolist()
            for i in range(height):
                tempArrRow[i] = iProcess.dct1d(tempArrRow[i])
            #applying dct1d column by column
            tempArrCol = np.array(tempArrRow)
            for j in range(width):
                tempArrCol[ :, j] = iProcess.dct1d(tempArrCol[ :, j])

            new_image.append(tempArrCol)
        return new_image


class iWrap:
    """
        This class contains wrappers.
    """
    def __init__(self):
        pass

    def color_clamped(img:list) -> list:
        """
            color_clamped(img)
            Takes image as input and returns normalized image as output.
        """
        return np.multiply(img, 255/np.max(img))

    def imageParser(img:list, row, col, channel:int=1, radius:int=1, padding=True) -> list:
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
        # image = np.pad(img, radius, mode='constant') if padding else img
        image = img
        sub_shape = (radius, radius)
        parsedCh = np.empty((channel), dtype=np.ndarray)
        for ch in range(channel):
            temp = tuple(np.subtract(image[ch].shape, sub_shape) + 1) + sub_shape
            temp2 = ast(image[ch], temp, image[ch].strides * 2)
            # parsedCh[ch] = (temp2.reshape((-1,) + sub_shape))
            parsedCh[ch] = temp2
        # print(parsedCh)
        return parsedCh
    
    def cvIm2List(img:list, radius=3, padding=True) -> list:
        """
            cvIm2List(image)

        Takes image read by cv2 as input and returns 3 layers of grayscale image as BGR.

        Parameters:
        --------------------------------
        image: list
        """
        img = np.pad(img, int((radius-1)/2), mode='constant') if padding else img
        (row, col, channel) = img.shape
        imgList = np.empty((channel), dtype=np.ndarray)
        for ch in range(channel):
            imgList[ch] = img[:,:,ch] 
        # print(imgList[0])
        return imgList

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

    inputImg = cv.imread(imPath)
    (height, width, channel) = inputImg.shape[:3]
    size = (height, width, channel)
    # print(size)
    # print(inputImg)

    # Create seperate windows for input image and output image
    cv.namedWindow(in_window_name)
    cv.namedWindow(out_window_name)

    # # Set mouse callback function for input image
    # cv.setMouseCallback("{}".format(in_window_name), iManipulate.draw_circle)

    # Show input image and wait for user input
    case = 0

    while 1:
        cv.imshow(in_window_name, inputImg)
        cv.waitKey(1)
        # Test output images
        if cv.waitKey(20) & 0xFF == 27:
            break


    # # Test output images
    # outputImg = iManipulate.translateIm(inputImg, size, 100, -100)
    # iManipulate.showIm(out_window_name, outputImg)
    # outputImgR = iManipulate.rotateIm(inputImg, size, 45)
    # iManipulate.showIm(out_window_name, outputImgR)
    # outputImgS = iManipulate.scaleIm(inputImg, size, .5, .5)
    # iManipulate.showIm(out_window_name, outputImgS)

    # Test output images

    # Original image
    # outputImg = iManipulate.processIm(inputImg, size[2], '')
    # cv.imshow(out_window_name, outputImg)
    # cv.waitKey(0)
    # case =+ 1

    # Gamma corrected image
    print(case)
    # outputImgG = iManipulate.processIm(inputImg, size[2], 'gamma', gamma=1.5)
    outputImgG = iProcess.gammaCorrection(inputImg, gamma=1.5)
    cv.imshow(out_window_name, outputImgG) 
    cv.imwrite("gammaCorrected.jpg", outputImgG)
    cv.waitKey(0)
    case =+ 1
    
    # Edge detected image using Prewitt operator (takes too long to process - not recommended to run)
    print(case)
    # outputImgP = iProcess.prewittEdgeDetection(inputImg)
    # cv.imshow(out_window_name, outputImgP)
    # cv.imwrite("edgeDetected.jpg", outputImgP)
    # cv.waitKey(0)
    case =+ 1
    
    # DCT of the image
    print(case)
    # outputImgD = iManipulate.processIm(inputImg, size[2], 'dct')
    outputImgD = iProcess.dct2d(inputImg)
    # cv.imshow(out_window_name, outputImgD)
    cv.imwrite("dct.jpg", outputImgD)
    cv.waitKey(0)

    
    print("Exiting...")



if __name__ == "__main__":
    main()


#TODO: Update input image when a marker is placed. Transform the markers to output image. Add these to GUI. 