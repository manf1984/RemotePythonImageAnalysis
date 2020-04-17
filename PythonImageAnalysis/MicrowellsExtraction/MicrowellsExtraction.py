'''
Created on 17 Apr 2020

@author: Andrea Manfrin
'''

import numpy as np
import skimage.measure
import skimage.filters as skf
import skimage.morphology as skm
from skimage.external.tifffile import TiffFile
from skimage.external.tifffile import TiffWriter
from pathlib import Path
import re
import sys
import timeit


#I can define the functions already used outside of the main function.

#This is the function that creates each single "square":
def returnSquare(coords, radius, refImage):
    y = np.uint16(np.round(coords[0]))
    x = np.uint16(np.round(coords[1]))
    cols = np.array([pntX for pntX in range(x-radius, x+radius+1,1) if ((pntX >= 0) and (pntX <= refImage.shape[1]-1))])
    rows = np.array([pntY for pntY in range(y-radius, y+radius+1,1) if ((pntY >= 0) and (pntY <= refImage.shape[0]-1))])
    return np.array(np.meshgrid(rows, cols))


#FROM HERE THE MAIN FUNCTION:
#The arguments are:
#1) the folder were the files to process are located (string version of the absolute path);
#2) the "radius" (in pixels) of the square used to crop the wells (edge of the square = 2*radius + 1);
#3) the minsize (= minimum area size in pixels) of the objects to keep;
#4) the maxsize (= maximum area size in pixels) of the objects to keep;
#5) Zproject (default "None"): if "None" (or whatever the other random word) the function will return
#   files with the Z-stcks are separated; if "max", "Max" or "MAX" it will return maximum Z-projections of the 
#   Z-stacks for each channel; if "mean", "Mean" or "MEAN" it will return mean Z-projections of the Z-stycks for 
#   each channel; if "min", "Min" or "MIN" it will return minimum Z-projections of the Z-stycks for
#   each channel
def processFiles(folder : str = "", radius : int = 330, minsize : int = 20000, maxsize : int = 350000, Zproject : str = None):
    folderPath = Path(folder)
    pattern = re.compile(r".*\.tif+$")
    fileList = [file for file in folderPath.iterdir() if file.is_file() and re.search(pattern, file.as_posix())]
    #print(fileList)    #Now I have the list of all ".tif" or ".tiff" files in PosixPath object format!
    
    #Now the loop starts! Before that I will define here everything that is common for each cycle of the "for loop":
    pattern = re.compile(r".tif+$")
    #Select a file from the list:
    imageCount = 0
    for file in fileList:
        startImageTime = timeit.default_timer()
        #Create a file-dedicated output folder/directory:
        outFolder = None
        mo = re.split(pattern, file.as_posix())
        outFolder = mo[0]
        #print(outFolder)
        Path(outFolder).mkdir(exist_ok = False)
        with TiffFile(file.as_posix()) as tif:
            imageInitial = tif.asarray()
        #Max project of all the fluo channels and all the Z-stacks in a single image.
        #Remember to place the 3 fluo channels as 2nd, 3rd and 4th and the BF as 1st in the original images.
        stackFluo = imageInitial[:, 1:4, :, :]
        stackFluo = stackFluo.max(axis = 0)
        stackFluo = stackFluo.max(axis = 0)
        
        #Filtering with gaussian filter:
        stackFluoFiltered = skf.gaussian(stackFluo, sigma = 10)

        #Find ideal threshold value
        #Threshold with "Li threshold method" (this takes a bit of time):
        threshold = skf.threshold_li(stackFluoFiltered)
        
        #Create a mask (boolean array of values that are greater than the threshold):
        stackFluoThreshold = stackFluoFiltered > threshold
        
        #Binary processing of mask -> Binary Dilation (using a disk of 7 pixels of diameter):
        circle = skm.disk(3)
        stackProcessed = skm.binary_dilation(stackFluoThreshold, selem = circle)
        
        #Object/particle identification + labeling of object/particles identified:
        stackLabels = skimage.measure.label(stackProcessed)
        
        #Create the collection of all the properties of all the identified objects/particles:
        objProperties = skimage.measure.regionprops(stackLabels)
        
        #Creates a list with all the squares centered on the centroids of all the desired objects (selected by area).
        #It makes use of the function "returnSquare" defined at the beginning of this section.
        #It takes radius, minsize and maxsize directly from the arguments of the "processFiles" function!
        squaresList = [returnSquare(obj.centroid, radius, stackFluo) for obj in objProperties if obj.area > minsize and obj.area < maxsize]
        
        #Here there is the part with the eventual Z-projection according to the method chosen:
        if Zproject == "max" or Zproject == "Max" or Zproject == "MAX":
            imageInitial = imageInitial.max(axis = 0)
        if Zproject == "min" or Zproject == "Min" or Zproject == "MIN":
            imageInitial = imageInitial.min(axis = 0)
        if Zproject == "mean" or Zproject == "Mean" or Zproject == "MEAN":
            imageInitial = np.rint(imageInitial.mean(axis = 0))
            imageInitial = imageInitial.astype(np.uint8)
        
        #Save the wells as separated images:
        counter = 0
        for square in squaresList:
            counter = counter + 1
            filename =""
            if counter<10:
                filename = "0" + str(counter) + ".tif"
            else:
                filename = str(counter) + ".tif"
    
            path = outFolder + "/" + filename
            
            if Zproject == "max" or Zproject == "Max" or Zproject == "MAX" or Zproject == "min" or Zproject == "Min" or Zproject == "MIN" or Zproject == "mean" or Zproject == "Mean" or Zproject == "MEAN": 
                image = imageInitial[:, square[0].min():square[0].max(), square[1].min():square[1].max()]
            else:
                image = imageInitial[:, :, square[0].min():square[0].max(), square[1].min():square[1].max()]
    
            with TiffWriter(path, bigtiff = False, imagej = True) as writer:
                writer.save(image)
                writer.close()
        finalImageTime = timeit.default_timer()
        imageCount = imageCount + 1
        print("Time for file ", imageCount, ": ", finalImageTime-startImageTime, " s\n")

                
totalStartTime = timeit.default_timer()
try:
    processFiles("/Users/andrea/Desktop/Data_Test_Segmenetation/MultipleFiles", radius = 330, minsize = 20000, maxsize = 350000, Zproject = None)
except FileNotFoundError:
    sys.stderr.write("Ehi you!! Provide as argument to the \"processFiles\" function a valid path to a folder!")
    raise
except FileExistsError:
    sys.stderr.write("One or more output folders seem to already exist in the input directory.\nRemove them or cut-and-paste them somewhere else and re-run the code.")
    raise
totFinalTime = timeit.default_timer()

print("Total Time:")
print(totFinalTime-totalStartTime, " s")
