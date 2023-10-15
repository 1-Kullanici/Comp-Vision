import argparse
import cv2 as cv
import numpy as np

class iManipulate:
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


    # mouse callback function
    def draw_circle(event,x,y,flags, param):

        if event == cv.EVENT_LBUTTONDOWN:
            pass

        elif event == cv.EVENT_LBUTTONUP:
            cv.circle(img,(x,y),5,(0,0,255),-1)  ########### There is a problem with this line. It does not take img as input. ##############################
            cursorIndex.append([x,y])
            print(cursorIndex)



    def translateIm(img, size, tx=0, ty=0):

        tM = np.float32([[1, 0, tx], [0, 1, ty]]) # Translation matrix
        newImg = cv.warpAffine(img, tM, (size[1], size[0]))
        
        return newImg


    def scaleIm(img, size, scale_x=1, scale_y=1):

        tM = np.float32([[scale_x, 0, 0], [0, scale_y, 0]]) # Translation matrix
        newImg = cv.warpAffine(img, tM, (size[1], size[0]))
        
        return newImg


    def rotateIm(img, size, angle=0):

        tM = cv.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1)
        newImg = cv.warpAffine(img, tM, (size[1], size[0]))
        return newImg


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