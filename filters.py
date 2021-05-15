import cv2 as cv
import numpy as np
import os
import shutil
import glob
img = cv.imread('photos/take.jpg')

#resizing
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (width, height)
output = cv.resize(img, dsize)
# cv.imshow('resize',output)

#blackandwhite
gray = cv.cvtColor(output,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

#rgbeffect
rgb = cv.cvtColor(output,cv.COLOR_BGR2RGB)
# cv.imshow('rgb',rgb)

#hsv
#hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)
#cv.imshow('HSV', hsv)

#lab
#lab = cv.cvtColor(output, cv.COLOR_BGR2LAB)
#cv.imshow('LAB', lab)

#blur
blur = cv.GaussianBlur(output, (7,7), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

#dilating
dilated = cv.dilate(output, (7,7), iterations=2)
# cv.imshow('Dilated', dilated)

#canny
canny = cv.Canny(output, 125, 175)
# cv.imshow('Canny Edges', canny)

#thresholding
threshold, thresh = cv.threshold(output, 100, 150, cv.THRESH_BINARY )
# cv.imshow('Simple Thresholded', thresh)

#watercolor

gray_1 = cv.medianBlur(gray, 5)
edges = cv.adaptiveThreshold(gray_1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 5)

bilateral = cv.bilateralFilter(output, d=9, sigmaColor=200,sigmaSpace=200)
#cv.imshow('bil_blur',bilateral)

cartoon = cv.bitwise_and(bilateral, bilateral, mask=edges)
# cv.imshow('cartoon',cartoon)

#pencilsketch

invert = cv.bitwise_not(gray)
#cv.imshow('invert',invert)

smoothing = cv.GaussianBlur(invert, (21, 21),sigmaX=0, sigmaY=0)
##cv.imshow('smoothing',smoothing)

def dodgeV2(x, y):
    return cv.divide(x, 255 - y, scale=256)
pencilsketch = dodgeV2(gray, smoothing)
# cv.imshow('pencilsketch',pencilsketch)

#saving iamges temporarily
os.mkdir("temp")
cv.imwrite("temp/output.jpg", output)
cv.imwrite("temp/blur.jpg", blur)
cv.imwrite("temp/rgb.jpg", rgb)
cv.imwrite("temp/canny.jpg", canny)
cv.imwrite("temp/thresh.jpg", thresh)
cv.imwrite("temp/gray.jpg", gray)
cv.imwrite("temp/dilated.jpg", dilated)
cv.imwrite("temp/cartoon.jpg", cartoon)
cv.imwrite("temp/pencilsketch.jpg", pencilsketch)

#series of images as slideshow

def process():
    path = "temp"
    filenames = glob.glob(os.path.join(path, "*"))

    prev_image = np.zeros((500, 500, 3), np.uint8)
    for filename in filenames:
        print(filename)
        img = cv.imread(filename)

        height, width, _ = img.shape
        if width < height:
            height = int(height*500/width)
            width = 500
            img = cv.resize(img, (width, height))
            shift = height - 500
            img = img[shift//2:-shift//2,:,:]

        else:
            width = int(width*500/height)
            height = 500
            shift = width - 500
            img = cv.resize(img, (width, height))
            img = img[:,shift//2:-shift//2,:]

        for i in range(101):
                    alpha = i/100
                    beta = 1.0 - alpha
                    dst = cv.addWeighted(img, alpha, prev_image, beta, 0.0)

        cv.imshow("Slideshow", dst)
        if cv.waitKey(1) == ord('q'):
            return

        prev_image = img

        if cv.waitKey(1000) == ord('q'):
            return


process()



shutil.rmtree("temp", ignore_errors=False, onerror=None)