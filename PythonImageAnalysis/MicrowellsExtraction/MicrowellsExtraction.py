'''
Created on 17 Apr 2020

@author: Andrea Manfrin
'''

# IMPORTANT NOTES:
# It is better to run the "processFiles" function with the "Zproject" argument set to "None" = NO Z-Projection.
# This is because the code on the Segmentation expect files with all the Z-stack separated!


import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import skimage.filters as skf
import skimage.morphology as skm
from skimage.external.tifffile import TiffFile
from skimage.external.tifffile import TiffWriter
from pathlib import Path
import re
import sys
import timeit
import math


#This is the function that creates each single "square".
#It takes the coordinates of the centroid of the fluorescent object in each microwell and adds the "radius" (this is
#half of the edge of the square) to its coordinates, then creates the array of coordinates of the square around 
#the centroid:
def returnSquare(coords : tuple, radius : int, refImage : np.ndarray)-> np.ndarray:
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
    fileList.sort()
    #print(fileList)    #Now I have the list of all ".tif" or ".tiff" files in PosixPath object format!
    
    #Now the loop starts! Before that, I will define here everything that is common for each cycle of the "for loop":
    pattern = re.compile(r".tif+$")
    #Select a file from the list:
    imageCount = 0
    for file in fileList:
        startImageTime = timeit.default_timer()
        #Create a file-dedicated output folder/directory:
        outFolder = None
        mo = re.split(pattern, file.as_posix())
        outFolder = mo[0]
        #Here I substitute eventual spaces " " with underscores "_" in the last part of outfolder string
        #(the part corresponding to the filename):
        modFileName = None
        pattern2 = re.compile(r"/")
        mo = re.split(pattern2, outFolder)
        modFileName = mo[-1].replace(" ", "_")
        #Let's reset "outFolder" to an empty string and rebuild its parts of the path using the new
        #"mo" object, without the last part. That part will be added in the modified form at the end.
        outFolder = ""
        for el in mo[1:-1]:
            outFolder = outFolder + f"/{el}"
        outFolder = outFolder + f"/{modFileName}"
        #print(outFolder)
        
        #Create the actual folder:
        Path(outFolder).mkdir(exist_ok = False)
        
        #From here open the single file (one file per "for" loop cycle):
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
        stackLabels = label(stackProcessed)
        
        #Create the collection of all the properties of all the identified objects/particles:
        objProperties = regionprops(stackLabels)
        
        
        #HERE I HAVE TO:
        #1) KEEP ONLY THE PARTICLES THAT HAVE AN AREA COMPRISED BETWEEN THE "minsize" AND "maxsize" ARGUMENTS OF THE
        #   OF THE "processFiles" FUNCTION.
        #2) CREATE A LIST OF ALL THE CENTROIDS (LIST OF TUPLES OF INTS)
        originalCentroids = [obj.centroid for obj in objProperties if obj.area > minsize and obj.area < maxsize]
        
        #3) CALCULATE THE DISTANCE BETWEEN EACH OF THE CENTROIDS (USING PITAGORA'S THEOREM) AGAINST ALL THE OTHERS 
        #   (AVOID REPEATING COMPARISONS = EACH POINT WILL BE COMPARED ONLY WITH THE ONES THAT LIE DOWNSTREAM TO IT)
        #4)  IF POINTS THAT HAVE A DISTANCE LESS THAN 1/3*RADIUS FROM THE CURRENTLY ANALYZED POINT
        #    ARE FOUND TWO ASSOCIATED TYPES OF LISTS MUST BE CREATED (MAYBE ASSCOCIATED INSIDE A DICTIONARY?):
        #     a) A LIST WITH THE COORDINATES OF THE ANALYZED POINT AND THE COORDINATES OF ALL THE POINTS THAT HAVE
        #       BEEN FOUND CLOSE TO IT.
        #     b) A LIST WITH THE INDEX OF ECAH OF THIS POINT IN THE ORIGINAL LIST OF CENTROIDS.
        listOfCentroidsDicts = list() #THIS IS THE LIST THAT YOU HAVE TO USE IN THE NEXT STEP!!
        for ind in range(0, len(originalCentroids)-1, 1):
            dictCentroids = {"Points" : [], "Indeces" : []}
            indexPnt = ind
            pntX = originalCentroids[ind][0]
            pntY = originalCentroids[ind][1]
            
            for otherInd in range(indexPnt+1, len(originalCentroids), 1):
                distance = math.sqrt((pntX - originalCentroids[otherInd][0])**2 + (pntY - originalCentroids[otherInd][1])**2)
                if distance < int(round(radius)):
                    dictCentroids["Points"].append(originalCentroids[otherInd])
                    dictCentroids["Indeces"].append(otherInd)
            if len(dictCentroids["Indeces"]) > 0:
                dictCentroids["Points"].append(originalCentroids[ind])
                dictCentroids["Indeces"].append(indexPnt)
            listOfCentroidsDicts.append(dictCentroids)

        #5) NOW A CHECK MUST BE PERFORMED. INDEED ONE POINT COULD BE CLOSE TO MORE THAN ONE POINT...
        #   THIS CREATES A BIG PROBLEM IN THE PROCEDURE, BECAUSE SOME POINTS WILL BE REPEATED IN THE LISTS.
        #   THIS MUST BE AVOIDED.
        #   THE TRICK IS TO CHECK ALL THE OBTAINED LIST WITH THE INDECES AND SEE IF THEY SHARE AN INDEX.
        #   IF THIS IS THE CASE THE LISTS (BOTH COORDINATES AND INDECES) MUST BE COMBINED. THE REPEATED CENTROIDS'
        #   COORDINATES AND INDECES MUST BE COMBINED. THIS CAN BE OBTAINED BY TRANSFORMING THE LISTS IN SETS AND THEN 
        #   BACK TO LISTS.
        newlistOfCentroidsDicts = []
        indecesElemsToRemove = []
        if len(listOfCentroidsDicts) > 0:
            for ind in range(0, len(listOfCentroidsDicts)-1, 1):
                tempDict = listOfCentroidsDicts[ind]
                for incr in range(1, len(listOfCentroidsDicts)-ind, 1):
                    for el in listOfCentroidsDicts[ind]["Indeces"]:
                        if el in listOfCentroidsDicts[ind+incr]["Indeces"]:
                            tempDict["Points"].extend(listOfCentroidsDicts[ind+incr]["Points"])
                            tempDict["Indeces"].extend(listOfCentroidsDicts[ind+incr]["Indeces"])
                            indecesElemsToRemove.append(ind+incr)
                            indecesElemsToRemove.append(ind)
                if tempDict != listOfCentroidsDicts[ind]:
                    tempDict["Points"] = set(tempDict["Points"])
                    tempDict["Points"] = list(tempDict["Points"])
                    tempDict["Indeces"] = set(tempDict["Indeces"])
                    tempDict["Indeces"] = list(tempDict["Indeces"])
                    newlistOfCentroidsDicts.append(tempDict)
        
        #Convert the elements on the Indeces to remove to 0:
        if len(indecesElemsToRemove) > 0:
            for ind in indecesElemsToRemove:
                listOfCentroidsDicts[ind] = 0
                
        #Remove every element in the list that is equal to 0:
        while(True):
            try:
                listOfCentroidsDicts.remove(0)
            except ValueError:
                break
        #Append the groups of points 
        listOfCentroidsDicts.extend(newlistOfCentroidsDicts) #OK!! "listOfCentroidsDicts" IS THE FUCKING LIST WITH ALL
                                                             #THE GROUPS OF POINTS THAT ARE CLOSE ONE TO THE OTHER!!!!
        
        #6) FINAL STEP IS TO USE THE COORDINATES IN EACH COORDINATES LIST TO CALCULATE AN "AVEARGE" CENTROIDS = CALCULATE
        #   THE MEAN OF THE X AND Y COORDINATES FOR EACH POINT IN THE LIST. THIS WILL GIVE THE NEW CENTROID.
        #   APPEND THIS NEW CENTROID TO THE ORIGINAL CENTROIDS LIST.
        #   DELETE ALL THE ORIGINAL CENTROIDS USED TO CALCULATE THE "AVERAGE CENTROIDS" FROM THE ORIGINAL CENTROIDS LIST.
        #   TO DO THIS USE THE LIST WITH THE INDECES (= DELETE ELEMENTS OCCUPYING THOSE POSITIOINS IN THE ORIGINAL LIST).
        
        #I start with removing the centroids that have been included in groups of "close-enough centroids":
        for d in listOfCentroidsDicts:
            for ind in d["Indeces"]:
                originalCentroids[ind] = 0
        
        finalCentroids = set(originalCentroids)
        finalCentroids = list(finalCentroids)
        try:
            finalCentroids.remove(0)
        except ValueError:
            pass
        
        #Here I calculate the new "average" centroid for each "group of centroids" in "listOfCentroidsDicts".
        #And I append to "finalCentroids"
        for d in listOfCentroidsDicts:
            sumX = 0
            sumY = 0
            pointsNumber = len(d["Points"])
            if pointsNumber == 0:
                continue
            for pnt in d["Points"]:
                sumX += pnt[0]
                sumY += pnt[1]
            newX = int(round(sumX/pointsNumber))
            newY = int(round(sumY/pointsNumber))
            newPnt = (newX, newY)
            finalCentroids.append(newPnt)
        
        #NOW YOU CAN USE THIS MODIFIED VERSION OF THE LIST AS THE FIRST ARGUMENT OF THE "returnSquare" FUNCTION!    
        
        #Create a list with all the squares centered on the centroids of all the desired objects (selected by area).
        #Make use of the function "returnSquare" defined at the beginning of this section.
        #The function takes radius, minsize and maxsize directly from the arguments of the "processFiles" function!
        squaresList = [returnSquare(centroid, radius, stackFluo) for centroid in finalCentroids]
        
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
    processFiles("/path/to/your/folder", radius = 330, minsize = 20000, maxsize = 350000, Zproject = None)
except FileNotFoundError:
    sys.stderr.write("Ehi you!! Provide as argument to the \"processFiles\" function a valid path to a folder!")
    raise
except FileExistsError:
    sys.stderr.write("One or more output folders seem to already exist in the input directory.\nRemove them or cut-and-paste them somewhere else and re-run the code.")
    raise
totFinalTime = timeit.default_timer()

print("Total Time:")
print(totFinalTime-totalStartTime, " s")


