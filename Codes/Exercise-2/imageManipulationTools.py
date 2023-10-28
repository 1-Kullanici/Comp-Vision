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


    # def manipulateIm(image:list, tx:float=0, ty:float=0, scale_x:float=1, scale_y:float=1, angle:float=0, channel:int=1) -> list:
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
    #     pass


    def processIm(img:list, channel:int=1, mode:str='', radius:int=1, customKernel:list=[1], padding=True) -> list:
        """
            processIm(image, channel=1, mode='', radius=1, customKernel=[1])
            
        Takes image, channel, mode, kernel radius, custom kernel, and padding as input and returns the processed image.\n
        If channel is 1 (set by default), it returns a grayscale image. If channel is 3, it returns a color image.\n
        If no mode is selected, it returns the image as it is.\n
        If padding is enabled, the output image will not clipped.
        
        The processing operators are:
        --------------------------------
        - Gamma correction (point operation)
        - Prewitt edge detection (neighborhood operation)
        - DCT (discrete cosine transform) (global operation)
        
        Parameters:
        --------------------------------
        image: list
        channel: int
        mode: str
        radius: int
        customKernel: list
        padding: bool
        """
        kernel = iManipulate.kernelWrapper(mode, radius, customKernel)
        image  = iManipulate.imageParser(img, channel, radius, padding)
        processedCh = np.empty(radius)
        for ch in channel:
            multiplied_subs = np.einsum('ij,ijkl->ijkl',kernel,image[ch])
            processedCh[ch] = np.sum(np.sum(multiplied_subs, axis = -radius), axis = -radius)
        
        return processedCh


    def kernelWrapper(mode:str='', radius:int=1, customKernel:list=[1]) -> list:
        """
            kernelWrapper(mode, radius=1, customKernel=[1])
            
        Takes mode, radius, and custom kernel as input and returns the desired kernel as a square list.\n
        Mode allows a default kernel to be selected. If mode is 'custom', customParameters must be given.\n
        TODO: Mode can be 'average', 'gaussian', 'laplacian', 'prewitt', 'sobel', 'roberts', 'scharr', 'custom', or 'none'.\n
        If no mode is selected or a custom mode is selected but no custom kernel is provided, it returns a kernel that is [1].
        
        The pre-defined kernels are:
        --------------------------------
        - Gamma correction (point operation)
        - Prewitt edge detection (neighborhood operation)
        - DCT (discrete cosine transform) (global operation)
        
        Parameters:
        --------------------------------
        mode: str
        radius: int
        customKernel: list
        """
        defaultKernel = {  

            '...Gamma':  np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), # I may add radius to change size
            '...Prewitt':np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), # I may add radius to change size
            '...DCT':    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # I may add radius to change size
            # TODO: Define more deault kernels

        }
        if mode == 'custom':
            return customKernel
        else:
            try:
                return defaultKernel[mode]
            except:
                print("No mode is selected or the selected mode is not defined. Continuing with inactive kernel.")
                return [1]
        


    def imageParser(img:list, channel:int=1, radius:int=1, padding=True) -> list:
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


    # def ...(img:list, channel:int=1) -> list:
    #     pass



class iManipulateCv:
    """
        This class loads, shows, and manipulates images using OpenCV.
    """
    def __init__(self):

        global cursorIndex
        cursorIndex = []


    def loadIm(path):
        img = cv.imread(path) 
        # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img, img.shape


    def showIm(img, window_name="image"):
        cv.imshow(img, window_name)

        # waits for user to press any key (this is necessary to avoid Python kernel form crashing)
        cv.waitKey(0)


    def translateIm(img, size, tx=0, ty=0, scale_x=1, scale_y=1, angle=0): # Don't use scale and angle

        tM = np.float32([[1, 0, tx], [0, 1, ty]]) # Translation matrix
        newImg = cv.warpAffine(img, tM, (size[1], size[0]))
        
        return newImg


    def scaleIm(img, size, tx=0, ty=0, scale_x=1, scale_y=1, angle=0): # Don't use tx, ty, and angle

        tM = np.float32([[scale_x, 0, 0], [0, scale_y, 0]]) # Translation matrix
        newImg = cv.warpAffine(img, tM, (size[1], size[0]))
        
        return newImg


    def rotateIm(img, size, tx=0, ty=0, scale_x=1, scale_y=1, angle=0): # Don't use tx, ty, and scale

        tM = cv.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1)
        newImg = cv.warpAffine(img, tM, (size[1], size[0]))
        return newImg


    # # mouse callback function
    # def draw_circle(event,x,y,flags, param):

    #     if event == cv.EVENT_LBUTTONDOWN:
    #         pass

    #     elif event == cv.EVENT_LBUTTONUP:
    #         cv.circle(img,(x,y),5,(0,0,255),-1)  ########### There is a problem with this line. It does not take img as input. ##############################
    #         cursorIndex.append([x,y])
    #         print(cursorIndex)


class iManipulateQt:
    """
        This class manipulates images using PyQt.
    """
    def __init__(self):
        pass


    def translateIm(img, size, tx=0, ty=0, scale_x=1, scale_y=1, angle=0): # Don't use scale and angle
        pass


    def scaleIm(img, size, tx=0, ty=0, scale_x=1, scale_y=1, angle=0): # Don't use tx, ty, and angle
        pass


    def rotateIm(img, size, tx=0, ty=0, scale_x=1, scale_y=1, angle=0): # Don't use tx, ty, and scale
        pass


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