'''
Created on 5 Jul 2020

@author: Andrea Manfrin
'''

#Import of necessary Modules and Packages:
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
from skimage.measure import label, regionprops
import skimage.filters as skf
import skimage.morphology as skm
from cellpose.models import Cellpose
import math
from scipy import stats
from tifffile import TiffFile
from tifffile import TiffWriter
from scipy.ndimage.morphology import binary_fill_holes as fillHoles
import timeit


#This class does the work of collecting all the parameters desired on the aggregates, for all
#the aggregates of a sinlge picture (=well). The values of the parameters are collected in lists that
#can be retrieved using the dedicated get methods.
class AnalyzeWell(object):
    def __init__(self, inputFolder : str, stackMethod : str = "Max") -> None:
        #The folder containing the files used for the segmentation (with all the channels and Z-stcks separated 
        #and not segmented):
        self.inputFolder = inputFolder
        #This can be "max"/"Max"/"MAX", "min"/"Min"/"MIN", "mean"/"Mean"/"MEAN" or "median"/"Median"/"MEDIAN".
        #It is the method used to project all the Z-stacks into one:
        self.stackMethod = stackMethod
        
        #Folder from which to take the segmentation Masks:
        self.maskFolder = self.inputFolder + "/Masks"
        
        #Extraction of .tif and/or .tiff files paths from the maskFolder and creation of a fileList.
        #To convert these "file Path object" into a string you have to call the .as_posix() method on them.
        pattern = re.compile(r"^.*\.tif+$")
        folderPath = Path(self.maskFolder)
        self.fileList = [file.as_posix() for file in folderPath.iterdir() if file.is_file() and re.search(pattern, file.as_posix())]
        self.fileList.sort()
        
        
        #Here are the lists where I will store the values of the parameters extracted from each single aggregate:
        self.areaList = []  #list with area values
        self.lengthList = []  #list with length values
        self.eccList =[]  #list with eccentricity values
        self.roundList = []  #list with roundness values
        #Below follow the lists containing values for the signals on the different channels. In this case
        #SOX17 (Original channel = 4), BRA (Original channel = 2), SOX2 (Original channel = 3): 
        self.totSignalSOX17 = [] #sum of all the fluo signal contained in the aggregate for this gene.
        self.signalPerUnitSOX17 = [] #average signal per aggregate area unit (= per pixel) for this gene.
        self.occupancySOX17 = [] #percentage of aggregate area covered by pixel positive for this gene (defined using Li threshold)
        self.totSignalSOX2 = [] #sum of all the fluo signal contained in the aggregate for this gene.
        self.signalPerUnitSOX2 = [] #average signal per aggregate area unit (= per pixel) for this gene.
        self.occupancySOX2 = [] #percentage of aggregate area covered by pixel positive for this gene (defined using Li threshold)
        self.totSignalBRA = [] #sum of all the fluo signal contained in the aggregate for this gene.
        self.signalPerUnitBRA = [] #average signal per aggregate area unit (= per pixel) for this gene.
        self.occupancyBRA = [] #percentage of aggregate area covered by pixel positive for this gene (defined using Li threshold)
        
        
        #Start the process for each file:
        for file in self.fileList:
            #Load the mask as an numpy ndarray: 
            mask = None
            with TiffFile(file) as tifMask:
                mask = tifMask.asarray()
                tifMask.close()
            
            #Load the original image corresponding to the loaded mask with all the channels and Z-stacks as a numpy ndarray:
            pattern02 = re.compile(r"/")
            mo = re.split(pattern02, file)
            originalPath = self.inputFolder + "/" + mo[-1]
            originalImage = None
            with TiffFile(originalPath) as tif:
                originalImage = tif.asarray()
                tif.close()
            
            #Project the Z-stacks of originalImage and delete the BF channel:DA SISTEMARE!!!
            if self.stackMethod == "max" or "Max" or "MAX":
                 originalImage = originalImage.max(axis = 0)
            elif self.stackMethod == "min" or "Min" or "MIN":
                originalImage = originalImage.min(axis = 0)
            elif self.stackMethod == "min" or "Min" or "MIN":
                originalImage = np.median(originalImage, axis = 0)
                originalImage = originalImage.astype(np.uint8)
            elif self.stackMethod == "meann" or "Mean" or "MEAN":
                originalImage = originalImage.mean(axis = 0)
                originalImage = np.around(originalImage, decimals = 0)
                originalImage = originalImage.astype(np.uint8)
            else:
                originalImage = originalImage.max(axis = 0)
            
            originalImage = originalImage[1:4, :, :]
            
            #Segment each channel of the originalImage with the mask of the corresponding file.
            #Just multiplying the originalImage for the mask you can do this. Even though the originalImage has 3 dimensions
            #and the mask only one, the operation with "mask" is broadcasted for all the channels of originalImage!
            segmentedImage = originalImage * mask
            
            #Let's get the geometrical properties of the aggregates directly from the mask. I identify particles on the mask
            #and this will result in one particle corresponsidng to the segmented area = to the aggregate!
            #Object/particle identification + labeling of object/particles identified:
            maskParticles = label(mask)  
            #Create the collection of all the properties of all the identified objects/particles:
            particlesProps = regionprops(maskParticles)
            #There is only one particle, in position "0":
            try:
                aggProps = particlesProps[0]
            except IndexError:
                continue
            if len(particlesProps) > 1:
                areas = [el.area for el in particlesProps]
                maxArea = max(areas)
                indexToKeep = areas.index(maxArea)
                aggProps = particlesProps[indexToKeep]
                    
                
            
            
            #Area in pixels:
            aggArea = aggProps.area
            self.areaList.append(aggArea)
            
            #I consider as Lentgh of the aggregate the major axis of the ellipse:
            aggLength = aggProps.major_axis_length
            self.lengthList.append(aggLength)
            
            #Eccentricity, a measure of elongation, when 0 it is the case of a circle:
            aggEccentricity = aggProps.eccentricity
            self.eccList.append(aggEccentricity)
            
            #My measure of elongation, I call it Roundness, it goes from 0 to 1, and 1 is equivalent to a circle.
            #The smaller is the Roundness the more elongated is the particle:
            aggRoundness = aggProps.minor_axis_length/aggLength
            self.roundList.append(aggRoundness)
    
#             print("Geometrical properties of the aggregate from picture ", mo[-1])
#             print("Area: ", aggArea, ",  Length: ", aggLength, ",  Eccentricity: ", aggEccentricity, ",  Roundness: ", aggRoundness)
            
            #Here I calculate and store the values of the parameters linked with the fluorescnet signals:
            segmentedImageSOX17 = segmentedImage[2, :, :]
            #Store this!
            self.totSignalSOX17.append(segmentedImageSOX17.sum())
            #Store this!
            self.signalPerUnitSOX17.append(self.totSignalSOX17[-1] / aggProps.area)
            thresholdSOX17 = skf.threshold_li(segmentedImageSOX17)
            thresholdedSOX17 = segmentedImageSOX17 > thresholdSOX17
            thresholdedSOX17 = thresholdedSOX17.astype(np.uint8)
            #Store this!
            self.occupancySOX17.append((thresholdedSOX17.sum() / aggProps.area) * 100)
            
            segmentedImageSOX2 = segmentedImage[1, :, :]
            #Store this!
            self.totSignalSOX2.append(segmentedImageSOX2.sum())
            #Store this!
            self.signalPerUnitSOX2.append(self.totSignalSOX2[-1] / aggProps.area)
            thresholdSOX2 = skf.threshold_li(segmentedImageSOX2)
            thresholdedSOX2 = segmentedImageSOX2 > thresholdSOX2
            thresholdedSOX2 = thresholdedSOX2.astype(np.uint8)
            #Store this!
            self.occupancySOX2.append((thresholdedSOX2.sum() / aggProps.area) * 100)
            
            segmentedImageBRA = segmentedImage[0, :, :]
            #Store this!
            self.totSignalBRA.append(segmentedImageBRA.sum())
            #Store this!
            self.signalPerUnitBRA.append(self.totSignalBRA[-1] / aggProps.area)
            thresholdBRA = skf.threshold_li(segmentedImageBRA)
            thresholdedBRA = segmentedImageBRA > thresholdBRA
            thresholdedBRA = thresholdedBRA.astype(np.uint8)
            #Store this!
            self.occupancyBRA.append((thresholdedBRA.sum() / aggProps.area) * 100)
    
    
    #Here are the methods to get the lists with the values for each parameter:
    def getArea(self):
        return self.areaList
    
    def getLength(self):
        return self.lengthList
    
    def getEccentricity(self):
        return self.eccList
    
    def getRoundness(self):
        return self.roundList
    
    def getTotSignalSOX17(self):
        return self.totSignalSOX17
    
    def getTotSignalSOX2(self):
        return self.totSignalSOX2
    
    def getTotSignalBRA(self):
        return self.totSignalBRA
    
    def getSignalUnitSOX17(self):
        return self.signalPerUnitSOX17
    
    def getSignalUnitSOX2(self):
        return self.signalPerUnitSOX2
    
    def getSignalUnitBRA(self):
        return self.signalPerUnitBRA
    
    def getOccupancySOX17(self):
        return self.occupancySOX17

    def getOccupancySOX2(self):
        return self.occupancySOX2
    
    def getOccupancyBRA(self):
        return self.occupancyBRA   


#Now the class that takes all the directory of the wells with the same condition and returns the full lists of values
#for the parameters of the aggregates.
#It makes use of the class "AnalyzeWell" and basically combine the values for each parameter of all the directories
#containing "segmented" aggergates that are prvided as a list to it.
class Condition (object):
    #Provide a list of strings each representing the path to a directory that you want to process as part of a same
    #experimental condtion
    def __init__(self, dirList : list, stackMethod : str = "Max") -> None:
        #This is the list of directories (expressed as string paths) that belong to the same experimental condition:
        self.dirList = dirList
        self.stackMethod = stackMethod
        
        #The lists that will contain the values of the various aggregates' parameters:
        self.areaList = []
        self.lengthList = []
        self.eccentricityList = []
        self.roundnessList = []
        self.totSignalSOX17 = []
        self.totSignalSOX2 = []
        self.totSignalBRA = []
        self.signalUnitSOX17 = []
        self.signalUnitSOX2 = []
        self.signalUnitBRA = []
        self.occupancySOX17 = []
        self.occupancySOX2 = []
        self.occupancyBRA = []
        #This list contain the name of the file from which each aggregate came. This allow to know from which
        #file (= picture = well) the values came and allow to eventually stratify the distribution on the
        #basis of the wells, just to see if the distribution is well-dependent (= the condition is eventaully not
        #experimetally robust):
        self.sourceList = []
    
        for directory in self.dirList:
            #Instatiate the "AnalyzeWell" object for each directory of dirList:
            well = AnalyzeWell(directory, self.stackMethod)
            
            #Add the elements of each list of "AnalyzeWell" to the corresponding lists of "Condition".
            #These lists will contain all the values for all the aggregates of all the directories:
            self.areaList.extend(well.getArea())
            self.lengthList.extend(well.getLength())
            self.eccentricityList.extend(well.getEccentricity())
            self.roundnessList.extend(well.getRoundness())
            
            self.totSignalSOX17.extend(well.getTotSignalSOX17())
            self.totSignalSOX2.extend(well.getTotSignalSOX2())
            self.totSignalBRA.extend(well.getTotSignalBRA())
            
            self.signalUnitSOX17.extend(well.getSignalUnitSOX17())
            self.signalUnitSOX2.extend(well.getSignalUnitSOX2())
            self.signalUnitBRA.extend(well.getSignalUnitBRA())
            
            self.occupancySOX17.extend(well.getOccupancySOX17())
            self.occupancySOX2.extend(well.getOccupancySOX2())
            self.occupancyBRA.extend(well.getOccupancyBRA())
            
            #Create a list with the labels of the file to add to "slef.source". This must should result for each
            #directory in labels all equal in anumber equal to the number of aggregates. This number can be extracted
            #from one of the other lists (e.g. well.areaList):
            pattern = re.compile(r"/")
            mo = re.split(pattern, well.inputFolder)
            fileName = mo[-1]
            labelList = [fileName] * len(well.getArea())
            self.sourceList.extend(labelList)
    
    #Here are the methods to get out the values of each parameter from the "Condition" object:
    def getArea(self):
        return self.areaList
    
    def getLength(self):
        return self.lengthList
    
    def getEccentricity(self):
        return self.eccentricityList
    
    def getRoundness(self):
        return self.roundnessList
    
    def getTotSignalSOX17(self):
        return self.totSignalSOX17
    
    def getTotSignalSOX2(self):
        return self.totSignalSOX2
    
    def getTotSignalBRA(self):
        return self.totSignalBRA
    
    def getSignalUnitSOX17(self):
        return self.signalUnitSOX17
    
    def getSignalUnitSOX2(self):
        return self.signalUnitSOX2
    
    def getSignalUnitBRA(self):
        return self.signalUnitBRA
    
    def getOccupancySOX17(self):
        return self.occupancySOX17
    
    def getOccupancySOX2(self):
        return self.occupancySOX2
    
    def getOccupancyBRA(self):
        return self.occupancyBRA
    
    def getSource(self):
        return self.sourceList

    
    
# #Instantiation of "AnalyzeWell" object on the files of the well "2019.09.23_CTRL01".
# #I use Max projection for Z-projection.
# well1 = AnalyzeWell("/Users/andrea/Desktop/Data_Test_Segmenetation/TRIAL/Already Processed/2019.09.23_CTRL01", "Max")

#Let's instatiate a "Condition" object with a list of 4 directories containing aggregates exposed to the "CTRL"
#condition:
CTRLList = ["/path/to/first/CTRL/folder",
            "/path/to/second/CTRL/folder"
            ]

CTRL = Condition(CTRLList, "Max")

#This is a "Condition" with 2 directories from the condition "Activin A"
ActAList = ["/path/to/first/ActA/folder",
            "/path/to/second/ActA/folder"
            ]

ActA = Condition(ActAList, "Max")


# #Visualize the results for "Condition" CTRL:
# fig1, ax = plt.subplots(2, 2)
# fig1.set_size_inches((10, 10))
# fig1.suptitle("CTRL")
# ax[0, 0].set_title("Area")
# ax[0, 0].hist(CTRL.getArea(), bins = 20)
# ax[0, 1].set_title("Length")
# ax[0, 1].hist(CTRL.getLength(), bins = 20)
# ax[1, 0].set_title("Eccentricity")
# ax[1, 0].hist(CTRL.getEccentricity(), bins = 20)
# ax[1, 1].set_title("Roundness")
# ax[1, 1].hist(CTRL.getRoundness(), bins = 20)
# fig1.tight_layout()
# plt.show();

# print(CTRL.getSource())

# #Visualize the results for "Condition" ActA:
# fig1, ax = plt.subplots(2, 2)
# fig1.set_size_inches((10, 10))
# fig1.suptitle("Activin A")
# ax[0, 0].set_title("Area")
# ax[0, 0].hist(ActA.getArea(), bins = 20)
# ax[0, 1].set_title("Length")
# ax[0, 1].hist(ActA.getLength(), bins = 20)
# ax[1, 0].set_title("Eccentricity")
# ax[1, 0].hist(ActA.getEccentricity(), bins = 20)
# ax[1, 1].set_title("Roundness")
# ax[1, 1].hist(ActA.getRoundness(), bins = 20)
# fig1.tight_layout()
# plt.show();

fig, ax = plt.subplots(2, 2)
fig.set_size_inches((8, 8))
fig.suptitle("CTRL vs. ActivinA - Aggregate Geometry", y = 1.03)
ax[0, 0].set_title("Area")
ax[0, 0].set_ylabel("Frequency")
ax[0, 0].set_xlabel("Area (pixels)")
sns.distplot(CTRL.getArea(), bins = 20, hist = True, kde = True, rug = False, ax = ax[0, 0], label = "Control", norm_hist = True)
sns.distplot(ActA.getArea(), bins = 20, hist = True, kde = True, rug = False, ax = ax[0, 0], label = "ActivinA", norm_hist = True)
ax[0, 1].set_title("Length")
ax[0, 1].set_ylabel("Frequency")
ax[0, 1].set_xlabel("Length (pixels)")
sns.distplot(CTRL.getLength(), bins = 20, hist = True, kde = True, rug = False, ax = ax[0, 1], label = "Control", norm_hist = True)
sns.distplot(ActA.getLength(), bins = 20, hist = True, kde = True, rug = False, ax = ax[0, 1], label = "ActivinA", norm_hist = True)
ax[1, 0].set_title("Eccentricity")
ax[1, 0].set_ylabel("Frequency")
ax[1, 0].set_xlabel("Eccentricity (a.u.)")
sns.distplot(CTRL.getEccentricity(), bins = 20, hist = True, kde = True, rug = False, ax = ax[1, 0], label = "Control", norm_hist = True)
sns.distplot(ActA.getEccentricity(), bins = 20, hist = True, kde = True, rug = False, ax = ax[1, 0], label = "ActivinA", norm_hist = True)
ax[1, 1].set_title("Roundness")
ax[1, 1].set_ylabel("Frequency")
ax[1, 1].set_xlabel("Roundness (a.u.)")
sns.distplot(CTRL.getRoundness(), bins = 20, hist = True, kde = True, rug = False, ax = ax[1, 1], label = "Control", norm_hist = True)
sns.distplot(ActA.getRoundness(), bins = 20, hist = True, kde = True, rug = False, ax = ax[1, 1], label = "ActivinA", norm_hist = True)
ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()
fig.tight_layout()
fig.savefig("/path/where/to/save/Geometrical_Features.pdf", dpi = 300, format = "pdf")
plt.show();


combinedListsforEccentricity = CTRL.getEccentricity()
combinedListsforEccentricity.extend(ActA.getEccentricity())
rightLimit = max(combinedListsforEccentricity)
leftLimit = min(combinedListsforEccentricity)
thresholds = np.linspace(leftLimit, rightLimit, 40)


fig1 = plt.figure(figsize = (4, 4))
ax1 = fig1.add_axes([0, 0, 1, 1])
sns.distplot(CTRL.getEccentricity(), hist = True, bins = thresholds, kde = False, rug = False, label = "Control", norm_hist = False, ax = ax1)
sns.distplot(ActA.getEccentricity(), hist = True, bins = thresholds, kde = False, rug = False, label = "ActivinA", norm_hist = False, ax = ax1)
ax1.legend()
plt.show();

sns.distplot(CTRL.getEccentricity(), hist = False, bins = thresholds, kde = True, rug = True, label = "Control", norm_hist = True)
sns.distplot(ActA.getEccentricity(), hist = False, bins = thresholds, kde = True, rug = True, label = "ActivinA", norm_hist = True)
plt.show();

t_test_ecc = stats.ttest_ind(CTRL.getEccentricity(), ActA.getEccentricity(), equal_var = False)
t_test_ecc

t_test_round = stats.ttest_ind(CTRL.getRoundness(), ActA.getRoundness(), equal_var = False)
t_test_round

t_test_area = stats.ttest_ind(CTRL.getArea(), ActA.getArea(), equal_var = False)
t_test_area

t_test_length = stats.ttest_ind(CTRL.getLength(), ActA.getLength(), equal_var = False)
t_test_length

mannwhitu_test_ecc = stats.mannwhitneyu(CTRL.getEccentricity(), ActA.getEccentricity(), alternative = "two-sided")
mannwhitu_test_ecc

mannwhitu_test_round = stats.mannwhitneyu(CTRL.getRoundness(), ActA.getRoundness(), alternative = "two-sided")
mannwhitu_test_round

mannwhitu_test_area = stats.mannwhitneyu(CTRL.getArea(), ActA.getArea(), alternative = "two-sided")
mannwhitu_test_area

mannwhitu_test_length = stats.mannwhitneyu(CTRL.getLength(), ActA.getLength(), alternative = "two-sided")
mannwhitu_test_length



fig2, ax2 = plt.subplots(1, 3)
fig2.set_size_inches((12, 4))
fig2.suptitle("CTRL vs. ActivinA - SOX17", y = 1.03)
ax2[0].set_title("Total Signal")
ax2[0].set_ylabel("Frequency")
ax2[0].set_xlabel("Signal (a.u.)")
sns.distplot(CTRL.getTotSignalSOX17(), bins = 20, hist = True, kde = True, rug = False, ax = ax2[0], label = "Control", norm_hist = True)
sns.distplot(ActA.getTotSignalSOX17(), bins = 20, hist = True, kde = True, rug = False, ax = ax2[0], label = "ActivinA", norm_hist = True)
ax2[1].set_title("Average Signal per Unit")
ax2[1].set_ylabel("Frequency")
ax2[1].set_xlabel("Signal (a.u.)")
sns.distplot(CTRL.getSignalUnitSOX17(), bins = 20, hist = True, kde = True, rug = False, ax = ax2[1], label = "Control", norm_hist = True)
sns.distplot(ActA.getSignalUnitSOX17(), bins = 20, hist = True, kde = True, rug = False, ax = ax2[1], label = "ActivinA", norm_hist = True)
ax2[2].set_title("Occupancy (%)")
ax2[2].set_ylabel("Frequency")
ax2[2].set_xlabel("Percentage of occupancy")
sns.distplot(CTRL.getOccupancySOX17(), bins = 20, hist = True, kde = True, rug = False, ax = ax2[2], label = "Control", norm_hist = True)
sns.distplot(ActA.getOccupancySOX17(), bins = 20, hist = True, kde = True, rug = False, ax = ax2[2], label = "ActivinA", norm_hist = True)
ax2[0].legend()
ax2[1].legend()
ax2[2].legend()
fig2.tight_layout(w_pad = 2)
fig2.savefig("/path/where/to/save/SOX17.pdf", dpi = 300, format = "pdf", bbox = "tight")
plt.show();


fig3, ax3 = plt.subplots(1, 3)
fig3.set_size_inches((12, 4))
fig3.suptitle("CTRL vs. ActivinA - SOX2", y = 1.03)
ax3[0].set_title("Total Signal")
ax3[0].set_ylabel("Frequency")
ax3[0].set_xlabel("Signal (a.u.)")
sns.distplot(CTRL.getTotSignalSOX2(), bins = 20, hist = True, kde = True, rug = False, ax = ax3[0], label = "Control", norm_hist = True)
sns.distplot(ActA.getTotSignalSOX2(), bins = 20, hist = True, kde = True, rug = False, ax = ax3[0], label = "ActivinA", norm_hist = True)
ax3[1].set_title("Average Signal per Unit")
ax3[1].set_ylabel("Frequency")
ax3[1].set_xlabel("Signal (a.u.)")
sns.distplot(CTRL.getSignalUnitSOX2(), bins = 20, hist = True, kde = True, rug = False, ax = ax3[1], label = "Control", norm_hist = True)
sns.distplot(ActA.getSignalUnitSOX2(), bins = 20, hist = True, kde = True, rug = False, ax = ax3[1], label = "ActivinA", norm_hist = True)
ax3[2].set_title("Occupancy (%)")
ax3[2].set_ylabel("Frequency")
ax3[2].set_xlabel("Percentage of occupancy")
sns.distplot(CTRL.getOccupancySOX2(), bins = 20, hist = True, kde = True, rug = False, ax = ax3[2], label = "Control", norm_hist = True)
sns.distplot(ActA.getOccupancySOX2(), bins = 20, hist = True, kde = True, rug = False, ax = ax3[2], label = "ActivinA", norm_hist = True)
ax3[0].legend()
ax3[1].legend()
ax3[2].legend()
fig3.tight_layout(w_pad = 2)
fig3.savefig("/path/where/to/save/SOX2.pdf", dpi = 300, format = "pdf", bbox = "tight")
plt.show();

fig4, ax4 = plt.subplots(1, 3)
fig4.set_size_inches((12, 4))
fig4.suptitle("CTRL vs. ActivinA - BRA", y = 1.03)
ax4[0].set_title("Total Signal")
ax4[0].set_ylabel("Frequency")
ax4[0].set_xlabel("Signal (a.u.)")
sns.distplot(CTRL.getTotSignalBRA(), bins = 20, hist = True, kde = True, rug = False, ax = ax4[0], label = "Control", norm_hist = True)
sns.distplot(ActA.getTotSignalBRA(), bins = 20, hist = True, kde = True, rug = False, ax = ax4[0], label = "ActivinA", norm_hist = True)
ax4[1].set_title("Average Signal per Unit")
ax4[1].set_ylabel("Frequency")
ax4[1].set_xlabel("Signal (a.u.)")
sns.distplot(CTRL.getSignalUnitBRA(), bins = 20, hist = True, kde = True, rug = False, ax = ax4[1], label = "Control", norm_hist = True)
sns.distplot(ActA.getSignalUnitBRA(), bins = 20, hist = True, kde = True, rug = False, ax = ax4[1], label = "ActivinA", norm_hist = True)
ax4[2].set_title("Occupancy (%)")
ax4[2].set_ylabel("Frequency")
ax4[2].set_xlabel("Percentage of occupancy")
sns.distplot(CTRL.getOccupancyBRA(), bins = 20, hist = True, kde = True, rug = False, ax = ax4[2], label = "Control", norm_hist = True)
sns.distplot(ActA.getOccupancyBRA(), bins = 20, hist = True, kde = True, rug = False, ax = ax4[2], label = "ActivinA", norm_hist = True)
ax4[0].legend()
ax4[1].legend()
ax4[2].legend()
fig4.tight_layout(w_pad = 2)
fig4.savefig("/path/where/to/save/BRA.pdf", dpi = 300, format = "pdf", bbox = "tight")
plt.show();


