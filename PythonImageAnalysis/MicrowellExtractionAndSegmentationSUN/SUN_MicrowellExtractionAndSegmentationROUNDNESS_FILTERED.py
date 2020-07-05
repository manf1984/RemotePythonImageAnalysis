'''
Created on 5 Jul 2020

@author: Andrea Manfrin (this script makes use of Cellpose -    doi:    https://doi.org/10.1101/2020.02.02.931238)
'''
#Import of necessary Modules and Packages:
import os
import re
import sys
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage.measure import label, regionprops
import skimage.filters as skf
import skimage.morphology as skm
from skimage.external.tifffile import TiffFile
from skimage.external.tifffile import TiffWriter
from scipy.ndimage.morphology import binary_fill_holes as fillHoles
from cellpose.models import Cellpose
import timeit


###############################################################################################################
####                                  MICROWELLS EXTRACTION SECTION                                        ####
###############################################################################################################
# IMPORTANT NOTES:
# For microwell extraction it is better to run the "processFiles" function with the "Zproject" argument set to 
#"None" = NO Z-Projection.
#This is because the code on the Segmentation expects files with all the Z-stack separated!


#This is a list that stores the coordinates (x,y) = (cols, rows) of each Top-Left corner of the cropped area
#for each microwell for each original file/well. This is used in the "segmentation" function to transpose the
#coordinates of each mask, so that they refer to the coordinates system of the whole initial BF image.
# This is what I have to do if I want to obtain an overlay of all the masks on top of the aggregates of
#the initial BF image.
listOfCornersByFolder = []
#For this same purpose you will also need to store the ordred list of paths (in str format) for each original image 
#file/well:
listOfWellPaths = []



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
        #Store in order the paths (in str format) for each original well-file. This is required to make the full overlay
        #of all the masks for all the files/wells on each original initial BF image.
        global listOfWellPaths
        listOfWellPaths.append(file.as_posix())
        
        #Start to work on the file.
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
        stackFluo = imageInitial[:, :]
        
        
        #Filtering with gaussian filter:
        stackFluoFiltered = skf.gaussian(stackFluo, sigma = 10)

        #Find ideal threshold value
        #Threshold with "Li threshold method" (this takes a bit of time):
        threshold = skf.threshold_li(stackFluoFiltered)
        
        #Create a mask (boolean array of values that are greater than the threshold):
        stackFluoThreshold = stackFluoFiltered < threshold
        
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
        
           ###################################################################################
           ####      HERE (POINT 2)IS WHERE I CAN INSERT THE FILTERING FOR OTHER          ####
           ####     GEOMETRICAL FEATURES LIKE CIRCULARITY, ROUNDNESS AND SO ON...         ####
           ###################################################################################
        #2) CREATE A LIST OF ALL THE CENTROIDS (LIST OF TUPLES OF INTS)
        #I will filter by Area (particles between minArea and maxArea values, in pixel number), but also by Roundness.
        #Roundness is defined as (4 * Area) / (pi * Major_Axis^2).
        #By some tests I performed on test-shapes I observed that this parameter is mostly sensitive to
        #elongation, but still it is a bit sensitive to the roundenss of the shape when tested on objects that
        #have the same level of "elongation". Since I think we are mostly interested in excluding objects that are
        #too much elongated and do not care about perfect roundness, I decide to keep the "particles" with a
        #Roundness >= 0.4:
        originalCentroids = [obj.centroid for obj in objProperties if obj.area > minsize and obj.area < maxsize and
                             (4*obj.area)/(math.pi*(obj.major_axis_length)**2) >= 0.4]
           

        #3) CALCULATE THE DISTANCE BETWEEN EACH OF THE CENTROIDS (USING PITAGORA'S THEOREM) AGAINST ALL THE OTHERS 
        #   (AVOID REPEATING COMPARISONS = EACH POINT WILL BE COMPARED ONLY WITH THE ONES THAT LIE DOWNSTREAM TO IT)
        #4)  IF POINTS THAT HAVE A DISTANCE LESS THAN RADIUS FROM THE CURRENTLY ANALYZED POINT
        #    ARE FOUND, TWO ASSOCIATED TYPES OF LISTS MUST BE CREATED (MAYBE ASSCOCIATED INSIDE A DICTIONARY?):
        #     a) A LIST WITH THE COORDINATES OF THE ANALYZED POINT AND THE COORDINATES OF ALL THE POINTS THAT HAVE
        #       BEEN FOUND CLOSE TO IT.
        #     b) A LIST WITH THE INDEX OF EACH OF THESE POINTS IN THE ORIGINAL LIST OF CENTROIDS.
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
        #   THE TRICK IS TO CHECK ALL THE OBTAINED LISTS WITH THE INDECES AND SEE IF THEY SHARE AN INDEX.
        #   IF THIS IS THE CASE THE LISTS (BOTH COORDINATES AND INDECES) MUST BE COMBINED. THE REPEATED CENTROIDS'
        #   COORDINATES AND INDECES MUST BE COMBINED. THEN BY TRANSFORMING THE LISTS IN SETS AND THEN 
        #   BACK TO LISTS IT IS POSSIBLE TO GET RID OF THE DUPLICATED ELEMENTS:
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
        #And I append the new centroids to "finalCentroids"
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
        
        #NOW YOU CAN FINALLY USE THIS MODIFIED VERSION OF THE LIST AS THE FIRST ARGUMENT OF THE
        #"returnSquare" FUNCTION!    
        
        #Create a list with all the squares centered on the centroids of all the desired objects (selected by area).
        #Make use of the function "returnSquare" defined at the beginning of this section.
        #The function takes radius, minsize and maxsize directly from the arguments of the "processFiles" function!
        squaresList = [returnSquare(centroid, radius, stackFluo) for centroid in finalCentroids]
        
        
        #Append to the "listOfCornersByFolder" list the list of coordinates (cols, rows) = (x, y) of the
        #Top-Left corner of each of these Squares in the squaresList so that it can then be used in the
        # "segmentation" function to shift the coordinates of each "mask" to align with the original aggregate
        # on the full initial BF image. This must be done for each file/well. So at the end I will have an ordered
        #list of tuples with each tuples representing the (y, x) coordinates of the corner of the cropping area of
        #each microwell for each original file!
        listOfCorners= []
        for square in squaresList:
            #Coordinate:
            y = 0
            x = 0
            
            #This is for the row-coordinates (y):
            y = square[0].min()
            
            #This is for the col-coordinates (x):
            x = square[1].min()
        
            listOfCorners.append((y, x))
        
        global listOfCornersByFolder
        listOfCornersByFolder.append(listOfCorners)
        
        
        #Here I will draw the coordinates of all the Squares on an image of the same size of the original one.
        #Then I will put it as a second layer in transparency on top of the original image.
        croppedAreasImage = np.ones((imageInitial.shape[-2], imageInitial.shape[-1], 4))
        figCrop, axCrop = plt.subplots(1)
        #Create a random color arrangement for the squares and the text on top of them.
        randomValueSquare = []
        randomValueText = []
        for ind in range(0, len(squaresList), 1):
            #Attribute a random value from 0-255 to the pixels of each square. This will result in different colors,
            #taken from the colormap "hsv":
            randomValue = np.random.randint(0, 256, size = 1, dtype = np.uint8)[0]
            randomValueSquare.append(randomValue)
            #This should make possible to have a color for text on the squares that is different enough from the color
            #of the background square:
            randomValueText.append(np.uint8(255 - randomValue))
        #This is the colormap I will use. I need to instantiate it because I want to get the color values (rgba 0-1)
        #for the text (starting from the list of random values of "randomValueText")
        nipy_spectral_cm = matplotlib.cm.get_cmap("nipy_spectral") 
        #Create the squares in the array and place the text objects on the image: 
        for ind, square in enumerate(squaresList, 0):
            croppedAreasImage[square[0], square[1], :] = nipy_spectral_cm(randomValueSquare[ind])
            text = ""
            if int(ind+1) < 10:
                text = "0" + str(ind+1)
            else:
                text = str(ind+1)
            axCrop.text(x = finalCentroids[ind][1], y = finalCentroids[ind][0], s = text, fontsize = 3,
                        color = nipy_spectral_cm(randomValueText[ind]), horizontalalignment = "center",
                        verticalalignment = "center")
        
        #Draw the original image and on top the array with the all the squares. The text objects have been already
        #inserted by the lines above:
        axCrop.imshow(imageInitial, cmap = "gray", vmin = 0, vmax = 255)
        axCrop.imshow(croppedAreasImage, alpha = 0.45)
        axCrop.set_axis_off()
        figCrop.savefig(outFolder + os.sep + "Cropped_Areas.pdf", dpi = 200, format = "pdf", bbox_inches = "tight")
        plt.show(figCrop);
        
        
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
                image = imageInitial[square[0].min():square[0].max(), square[1].min():square[1].max()]
            else:
                image = imageInitial[square[0].min():square[0].max(), square[1].min():square[1].max()]
    
            with TiffWriter(path, bigtiff = False, imagej = True) as writer:
                writer.save(image)
                writer.close()
        finalImageTime = timeit.default_timer()
        imageCount = imageCount + 1
        print("Time for file ", imageCount, ": ", finalImageTime-startImageTime, " s\n")
        


###############################################################################################################
####                                  AGGREGATES SEGMENTATION SECTION                                      ####
###############################################################################################################
###############################################################################################################
####                               THIS PART MUST BE CHECKED AND INTEGRATED                                ####
###############################################################################################################

def segmentation(mainFolder : str = ""):
    #Here all the folders with the images to segment are collected in the "foldersList" and are then returned one by one
    #to the code below (inside the "for loop"):
    mainPath = Path(mainFolder)
    foldersList = [folder.as_posix() for folder in mainPath.iterdir() if folder.is_dir()]
    foldersList.sort()

    for folderIndex, folder in enumerate(foldersList):
        #The folder containing the files to segment/to use for mask creation (use absolute path in string format):
        inputFolder = folder
        #The channel to use for mask creation. Use the BF channel.
        #It must be the same channel number (= in the same position) for all the files in the folder!!
        channelForSegm = 0

        #Timing the whole process:
        generalBeginning = timeit.default_timer()
        
        
        #Setting the deep-learning model of Cellpose to use for segmentation.
        #These settings are suggested:
        #myModel = Cellpose(gpu = False, model_type = "cyto", net_avg = True)
        #To shorten the processing time the "net_avg" argument can be set to False, but the segmentation will be less precise...
        myModel = Cellpose(gpu = False, model_type = "cyto", net_avg = True)


        #Create the output folders (TiffWriter cannot create folders by itself...):
        #Output folder for masks:
        maskOutput = inputFolder + "/Masks"
        os.mkdir(maskOutput)
        #Output folder for multipage .tif files with all the Channesl Max Z-projected + the mask as last channel:
        allChannelsAndMaskOutput = inputFolder + "/All_Channels_And_Mask"
        os.mkdir(allChannelsAndMaskOutput)
        #Output folder for "Original Image + Mask Overlay"
        overlayOutput = inputFolder + "/Mask_Overlay"
        os.mkdir(overlayOutput)


        #Extraction of .tif and/or .tiff files paths from the inputFolder and creation of a fileList.
        #To convert these "file Path object" into a string you have to call the .as_posix() method on them.
        pattern = re.compile(r"^.*\.tif+$")
        folderPath = Path(inputFolder)
        fileList = [file.as_posix() for file in folderPath.iterdir() if file.is_file() and re.search(pattern, file.as_posix())]
        fileList.sort()
        
        
#         #Create an empty image/array where you will place all the masks of each aggregates on top of the corresponding aggregate
#         #on the initial BF image.
#         global listOfWellPaths
#         segmentedAreasImage = None
#         originalBFImage = None
#         with TiffFile(listOfWellPaths[folderIndex]) as originalFile:
#             originalBFImage = originalFile.asarray()
#             segmentedAreasImage = np.ones((originalBFImage.shape[-2], originalBFImage.shape[-1], 4))
#             originalFile.close()
        
#         #These are the Figure and Axes instances used for each "folder/well" by imshow to represnet the
#         #overlay of masks on to the original BF image:
#         figSeg = None
#         axSeg = None
#         figSeg, axSeg = plt.subplots(1)
        
        
        #From here the "for loop" that will process each file in the inputFolder starts:
        for fileIndex, f in enumerate(fileList):
            #Timing for each single file:
            beginning = timeit.default_timer()

            #Here starts the hardcore part of the code!!!
            originalImage = None
            image = None
            with TiffFile(f) as tif:
                originalImage = tif.asarray()
            image = originalImage.copy()
            temp = image[:, :].copy()
            image = np.expand_dims(image, axis = 0)
            image = np.insert(image, obj = 0, values = temp[:, :], axis = 0)
            print(image.shape)


            #Set the paths (in string format) of the output files (it will create sub-folders containing files with the names of the original files):
            pattern = re.compile(r"/")
            mo = re.split(pattern, f)
            fileName = mo[-1]
            outputMask = inputFolder + "/Masks/" + fileName
            outputAllChannelsAndMask = inputFolder + "/All_Channels_And_Mask/" + fileName
            outputMaskOverlay = inputFolder + "/Mask_Overlay/" + fileName


            #This part will get the "measured radius" of the aggregate (in reality of the particle identified combining
            #all the fluo channels). The "measured radius" will be used to calculate a Segmentation Radius to use as argument
            #in the eval() method. One specific Segmentation Radius for each image!
            SegmentationRadius = None

            #Obtain an image with only the fluorescent channels, with the mean Z-projection of all the Z-stacks 
            #for each single channel and finally with all the fluo channels projected together on one single
            #2D-image using a max-projection:
            imageFluo = image[0, :, :]

            
            #Filtering with gaussian filter:
            imageFluo = skf.gaussian(imageFluo, sigma = 10)
            #Find ideal threshold value:
            #Threshold with "Li threshold method" (this takes a bit of time):
            threshold = skf.threshold_li(imageFluo)
            #Create a mask (boolean array of values that are greater than the threshold):
            imageFluo = imageFluo < threshold
            #Binary processing of mask -> Binary Dilation (using a disk of 7 pixels of diameter)
            #(Maybe this Dilation step is not necessary, but if you remove this step I think you will have to
            #re-adjust the ratio between "measuredRadius" and "SegmentationRadius", bringing it closer to 1.0):
            imageFluo = skm.binary_dilation(imageFluo, selem = skm.disk(3))
            #Object/particle identification + labeling of object/particles identified:
            particles = label(imageFluo)  
            #Create the collection of all the properties of all the identified objects/particles:
            particlesProps = regionprops(particles)

            area = None
            for el in particlesProps:
                if el.area > 20000:
                    area = el.area

            #This code converts the size of the "measuredRadius" (calculated here starting from "area") to 
            #the size of the "SegmentationRadius". This is made by multiplying "measuredRadius" 1,3 times.
            #1.3 is a value that I found empirically, checking which was the minimum best value for "
            #SegmentationRadius" of different particles of known "measuredRadius":
            if area != None:
                measuredRadius = math.sqrt(area/math.pi)
            else:
                continue
            SegmentationRadius = round((measuredRadius*1.3))


            #The code that follows mean-projects all the Z-stacks to a single one for each channel:
            image = np.rint(image.mean(axis = 0))
            image = image.astype(np.uint8)
            #image.shape

            #Here I run the Cellpose model "myModel" on the image using as "diameter" the value of "SegmentationRadius":
            segmentation = myModel.eval(image, channels = [0, 0], diameter = SegmentationRadius)

            #Extract the coordinates of the segmented object, what I call "mask", from the "segmentation" object:
            mask = segmentation[0][0]
            #Convert to boolean mask:
            mask = mask > 0
            #Convert the boolean "mask" to an np.uint8 "mask" (False = 0, True = 1):
            mask = mask.astype(np.uint8)
            #Creation of the BF only image segmented by Cellpose:
            image = image * mask


            #Unluckily Cellpose's segmentation identifies sometimes two or more "particles": one which contains the aggregate
            #located at the center of the image and sometimes some others located at the edges of the image.
            #I want to get rid of these evetual particles located at the edges of the images and keep only the
            #central particle.
            #I can use the centroid of each paticle to do that. 
            #Let's say that if one of the coordinates of the centroid of a particle are +/- 150 pixels from the center 
            #of the image, they are the one of the central particle and this must be kept. The other particles are not
            #considered, thus they are removed!
            imageXDim = image.shape[-1]
            imageYDim = image.shape[-2]
            imageCenterXY = {"x" : round(imageXDim/2), "y" : round(imageYDim/2)}
            #Transform "image" in a boolean mask where all the values different from 0 are set to True (= 1).
            imageThresholded = image > 0
            particles = label(imageThresholded)
            particlesProps = regionprops(particles)
            #Creation of the "mask" with only the coordinates to keep.
            #Keep only the coordinates of the "central" correct particle:
            coordsToKeep = None

            for el in particlesProps:
                y, x = el.centroid
                if (x < imageCenterXY["x"] + 150 and x > imageCenterXY["x"] - 150) or (y < imageCenterXY["y"] + 150 and y > imageCenterXY["y"] - 150):
                    coordsToKeep = el.coords

            #Sometimes Cellpose dose not manage to identify the cell aggregate... and it just gives some random shit
            #at the edges of the image... This makes it impossible to identify a "central particle"...
            #Thus, "coordsToKeep" remains None and when you iterate over it, it causes a blocking Exception.
            #For this reason it is better to catch this expcetion and say that if it appears the program should
            #skip (continue) the cycle for this file:

            mask = np.zeros(image.shape)
            try:
                for pnt in coordsToKeep:
                    mask[pnt[0], pnt[1]] = 1
            except(TypeError):
                print(f"Cellpose fucked it up for file: {f}. \n\n")
                continue

            #Mask the image with "mask" to keep only the correct particle in "image", the one containing the
            #aggregate:
            image = image * mask
            #Keep a copy of the image after Cellpose processing, to compare later how the rest of the code
            #improves the segmentation of the aggregates:
            imageCellpose = image.copy()
            imageCellpose = imageCellpose.astype(np.uint8)


            #From here I will apply to this "BF only image segmented by Cellpose" ("image") a series of
            #filters and size-based selection of particles to clean up further the segmented aggregate
            #from not-aggregated/shed cells deposited around it.

            #"FROM HERE FIRST ROUND:"
            #Apply an Hessian filter to "image". It is a filter that somehow show a sort of local variance.
            #It manages to somehow highlight a kind of border for the aggregates, more or less...
            filteredImage = skf.hessian(image, (2,2))
            #Invert the image (what is 0 will become 255 and viceversa):
            filteredImage = skimage.util.invert(filteredImage)
            #This step is required to cut out the border that the Hessian filter creates around the Cellpose segmented
            #area. This border is not good at all:
            filteredImage = filteredImage * mask
            #Boolean transformation of the "filteredImage" (every value which is greater than 0 it is set to True, meaning 1):
            filteredImage = filteredImage > 0
            #Erode "filteredImage" with a disk of 3 pixels of diameter:
            filteredImage = skm.binary_erosion(filteredImage, selem = skm.selem.disk(1))
            #Connect neighoboring particles using the "skm.closing" function (with default arguments):
            filteredImage = skm.closing(filteredImage)
            #Identify the particles created by the Hessian filter and all the other operations,
            #and create a collection of their properties:
            particles = label(filteredImage)
            particlesProps = regionprops(particles)
            #Keep only "big connected pieces" of the "filteredImage". In this way many of the marginal small pieces that
            #stay around the aggregate in the "shed cells area" are removed:
            minArea = 300   #This defines the minimum area (in pixels' number) for a particle to be kept in the image.
            selectedCoords = []
            for el in particlesProps:
                if el.area > minArea:
                    for pnt in el.coords:
                        selectedCoords.append(pnt)

            #Combination of all particles with area greater than "minArea" in the "finalMask":
            finalMask = np.zeros(image.shape)
            for pnt in selectedCoords:
                finalMask[pnt[0], pnt[1]] = 1

            #Closing using as selem a disk(25):"
            finalMask = skm.closing(finalMask, selem = skm.selem.disk(25))

            #Fill the holes (with a scipy function, check the "import section"):
            finalMask = fillHoles(finalMask) #THIS IS THE finalMask OBTAINED AFTER ALL THESE STEPS!!!



            #Mask with value of "0" or "1" for each pixel:
            segMaskImage = finalMask.copy()
            segMaskImage = segMaskImage.astype(np.uint8)
            
            
#             #Here I "insert" the mask of this file on the "segmentedAreasImage" using the right transposition of
#             #its coordinates. And I will do this for each microwell file. At the end I will create an "imshow" with
#             #the overaly of all the segmented areas on the original BF image. This for each folder (= original 
#             #well file). I will use the "listOfCornersByFolder" to know how to transpose each mask on the original
#             #BF image:
#             segMaskImageRGBA = np.ones((segMaskImage.shape[0], segMaskImage.shape[1], 4))
#             for rowIndex, row in enumerate(segMaskImage):
#                 for colIndex, col in enumerate(row):
#                     if col == 1:
#                         segMaskImageRGBA[rowIndex, colIndex] = [0.0, 0.0, 0.0, 0.0]
                    
#             segmentedAreasImage[listOfCornersByFolder[folderIndex][fileIndex][0]:listOfCornersByFolder[folderIndex][fileIndex][0] + segMaskImageRGBA.shape[0],
#                                 listOfCornersByFolder[folderIndex][fileIndex][1]:listOfCornersByFolder[folderIndex][fileIndex][1] + segMaskImageRGBA.shape[1],
#                                 :] = segMaskImageRGBA[:, :, :]
            
#             #Create lists of random colors for the masks and the associated text labels. The length of the lists
#             #equals the number of files for the current folder (= for the original file):
#             randomValueMask = []
#             randomValueText = []
#             for ind in range(0, len(fileList), 1):
#                 #Attribute a random value from 0-255 to the pixels of each square. This will result in different colors,
#                 #taken from the colormap "hsv":
#                 randomValue = np.random.randint(0, 256, size = 1, dtype = np.uint8)[0]
#                 randomValueMask.append(randomValue)
#                 #This should make possible to have a color for text on the squares that is different enough from the color
#                 #of the background square (less true for colors that occupy a central position in the colormap...):
#                 randomValueText.append(np.uint8(255 - randomValue))
            
#             #Get the colormap for the conversion of the above values to RGBA sequence:
#             nipy_spectral_cm = matplotlib.cm.get_cmap("nipy_spectral")
            

#             #Now give a color to the newly "inserted" mask. This will be done by searching for the "pixels" that have an
#             #RGBA value of [0.0, 0.0, 0.0, 0.0]
#             for rowIndex, row in enumerate(segmentedAreasImage):
#                 for colIndex, col in enumerate(row):
#                     if (col[0] == 0.0) and (col[1] == 0.0) and (col[2] == 0.0) and (col[3] == 0.0):
#                         segmentedAreasImage[rowIndex, colIndex] = nipy_spectral_cm(randomValueMask[fileIndex])
            
            
#             #Insert the text label for the current mask:
#             textYPosition = listOfCornersByFolder[folderIndex][fileIndex][0] + int(round(segMaskImage.shape[0]/2))
#             textXPosition = listOfCornersByFolder[folderIndex][fileIndex][1] + int(round(segMaskImage.shape[1]/2))
#             text = None
#             if int(fileIndex + 1) < 10:
#                 text = "0" + str(fileIndex + 1)
#             else:
#                 text = fileIndex + 1
            
#             axSeg.text(x = textXPosition, y = textYPosition, s = text, fontsize = 3,
#                         color = nipy_spectral_cm(randomValueText[fileIndex]), horizontalalignment = "center",
#                         verticalalignment = "center")
            
            
            
    #         #All the channels Z-Projected using Max projection + the final channel corresponding to the Mask in binary
    #         #form (0 and 1 only, as 8-bit integer). All combined in one multzpage ".tif" file per aggregate:
    #         combinedTifFile = originalImage.copy()
    #         combinedTifFile = combinedTifFile.max(axis = 0)
    #         combinedTifFile = np.insert(arr = combinedTifFile, obj = 4, values = segMaskImage, axis = 0)


            #Here the saving of all the files in ".tif" format:
            #Save the Mask:
            with TiffWriter(outputMask, bigtiff = False, imagej = True) as writer:
                writer.save(segMaskImage)
                writer.close()

    #         #Save the "combinedTifFile":
    #         with TiffWriter(outputAllChannelsAndMask, bigtiff = False, imagej = True) as writer:
    #             writer.save(combinedTifFile)
    #             writer.close()

            #Save BF from "originalImage" with "finalMask" overlaid:
            fig, ax = plt.subplots(1)
            originalImageTemp = originalImage.copy()
    #         originalImageTemp = originalImageTemp.max(axis = 0)
    #         originalImageTemp = originalImageTemp[0, :, :]
            ax.imshow(originalImageTemp, cmap = "gray")
            ax.imshow(finalMask, cmap = "inferno", alpha = 0.2)
            fig.savefig(outputMaskOverlay, format = "tif", dpi = 200)
            plt.close(fig)

            #Timing for each single file:
            end = timeit.default_timer()
            print(f"Time for processing the file: {end-beginning} s . \n\n")

        

        
        #Timing of the whole process:
        generalEnd = timeit.default_timer()
        print(f"Processing all the {len(fileList)} files took: {generalEnd-generalBeginning} s .")
        
#         #Create and save the overaid image where all the masks are overlaid to their respective aggregate on the 
#         #intial BF image of the original file (= well):
#         axSeg.imshow(originalBFImage, cmap = "gray", vmin = 0, vmax = 255)
#         axSeg.imshow(segmentedAreasImage, alpha = 0.45)
#         axSeg.set_axis_off()
#         figSeg.savefig(folder + os.sep + "Segmented_Areas.pdf", dpi = 200, format = "pdf", bbox_inches = "tight")
#         plt.show(figSeg);
        

###############################################################################################################
####                                     EXECUTION CODE FROM HERE                                          ####
###############################################################################################################
mainFolder = "/path/to/your/folder"


#Execution of microwell extraction:
totalStartTime = timeit.default_timer()
try:
    processFiles(mainFolder, radius = 300, minsize = 5000, maxsize = 400000, Zproject = None)
except FileNotFoundError:
    sys.stderr.write("Ehi you!! Provide as argument to the \"processFiles\" function a valid path to a folder!")
    raise
except FileExistsError:
    sys.stderr.write("One or more output folders seem to already exist in the input directory.\nRemove them or cut-and-paste them somewhere else and re-run the code.")
    raise
totFinalTime = timeit.default_timer()

print("Total Time:")
print(totFinalTime-totalStartTime, " s")


#Execution of segmentation:
segmentation(mainFolder)

