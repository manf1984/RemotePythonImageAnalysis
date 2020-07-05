'''
Created on 5 Jul 2020

@author: Andrea Manfrin  (this script makes use of Cellpose -    doi:    https://doi.org/10.1101/2020.02.02.931238)
'''

#Import of necessary Modules and Packages:
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.filters as skf
import skimage.morphology as skm
from cellpose.models import Cellpose
import math
from tifffile import TiffFile
from tifffile import TiffWriter
from scipy.ndimage.morphology import binary_fill_holes as fillHoles
import timeit


#Variables to define (this can be set by the user, especially inputFolder!!):
mainFolder = "path/to/your/folder"


#Here all the folders with the images to segment are collected in the "foldersList" and are then returned one by one
#to the code below (inside the "for loop"):
mainPath = Path(mainFolder)
foldersList = [folder.as_posix() for folder in mainPath.iterdir() if folder.is_dir()]
foldersList.sort()

for folder in foldersList:

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

    #From here the "for loop" that will process each file in the inputFolder starts:
    for f in fileList:
        #Timing for each single file:
        beginning = timeit.default_timer()

        #Here starts the hardcore part of the code!!!
        originalImage = None
        image = None
        with TiffFile(f) as tif:
            originalImage = tif.asarray()
        image = originalImage.copy()

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
        imageFluo = image[:, 1:4, :, :]
        imageFluo = imageFluo.mean(axis = 0).astype(np.uint8)
        imageFluo = imageFluo.max(axis = 0)

        #Filtering with gaussian filter:
        imageFluo = skf.gaussian(imageFluo, sigma = 10)
        #Find ideal threshold value:
        #Threshold with "Li threshold method" (this takes a bit of time):
        threshold = skf.threshold_li(imageFluo)
        #Create a mask (boolean array of values that are greater than the threshold):
        imageFluo = imageFluo > threshold
        #Binary processing of mask -> Binary Dilation (using a disk of 7 pixels of diameter)
        #(Maybe this Dilation step is not necessary, but if you remove this step I think you will have to
        #re-adjust the ratio between "measuredRadius" and "SegmentationRadius", bringing it closer to 1.0):
        imageFluo = skm.binary_dilation(imageFluo, selem = skm.disk(3))
        #Object/particle identification + labeling of object/particles identified:
        particles = skimage.measure.label(imageFluo)  
        #Create the collection of all the properties of all the identified objects/particles:
        particlesProps = skimage.measure.regionprops(particles)

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
        segmentation = myModel.eval(image[channelForSegm], channels = [0, 0], diameter = SegmentationRadius)

        #Extract the coordinates of the segmented object, what I call "mask", from the "segmentation" object:
        mask = segmentation[0][0]
        #Convert to boolean mask:
        mask = mask > 0
        #Convert the boolean "mask" to an np.uint8 "mask" (False = 0, True = 1):
        mask = mask.astype(np.uint8)
        #Creation of the BF only image segmented by Cellpose:
        image = image[channelForSegm] * mask


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
        particles = skimage.measure.label(imageThresholded)
        particlesProps = skimage.measure.regionprops(particles)
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
        particles = skimage.measure.label(filteredImage)
        particlesProps = skimage.measure.regionprops(particles)
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

        #All the channels Z-Projected using Max projection + the final channel corresponding to the Mask in binary
        #form (0 and 1 only, as 8-bit integer). All combined in one multzpage ".tif" file per aggregate:
        combinedTifFile = originalImage.copy()
        combinedTifFile = combinedTifFile.max(axis = 0)
        combinedTifFile = np.insert(arr = combinedTifFile, obj = 4, values = segMaskImage, axis = 0)


        #Here the saving of all the files in ".tif" format:
        #Save the Mask:
        with TiffWriter(outputMask, bigtiff = False, imagej = True) as writer:
            writer.save(segMaskImage)
            writer.close()

        #Save the "combinedTifFile":
        with TiffWriter(outputAllChannelsAndMask, bigtiff = False, imagej = True) as writer:
            writer.save(combinedTifFile)
            writer.close()

        #Save BF from "originalImage" with "finalMask" overlaid:
        fig, ax = plt.subplots(1)
        originalImageTemp = originalImage.copy()
        originalImageTemp = originalImageTemp.max(axis = 0)
        originalImageTemp = originalImageTemp[0, :, :]
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


