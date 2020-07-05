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
from skimage.measure._regionprops import RegionProperties
from cellpose.models import Cellpose
import math
from scipy import stats
from tifffile import TiffFile
from tifffile import TiffWriter
from scipy.ndimage.morphology import binary_fill_holes as fillHoles
import timeit
import pandas as pd


#This is the folder that contains all the folders with the files of the extracted microwells and segmented aggregates:
path = "/path/to/your/folder"
#Calculate the conversion factor  pixel/micrometers:
pixelLength = 9723.41/13442  #Length in micrometers of a pixel (along one dimension). Total width in micrometers/total width in pixels 
pixelArea = pixelLength**2  #Area, in square micrometers, of a pixel.


#This function takes a "mask" file path and returns the geometrical properties (only if the mask is composed of
#exactly 1 particle, otherwise it returns a None type object). It returns a tuple containing the file name (e.g "01") 
#as first element and a "skimage.measure._regionprops.RegionProperties" object as second element of the tuple!
def extractGeometryFromFile(file : str = ""):
    myFile = file
    mask = None
    with TiffFile(myFile) as tif:
        mask = tif.asarray()
    
    fileName = None
    listOfParticleProps = []
    
    sepPatt = re.compile(os.sep)
    tifPatt = re.compile(r".tif+$")
    mo1 = re.split(sepPatt, myFile)
    mo2 = re.split(tifPatt, mo1[-1])
    fileName = mo2[0]
    
    maskParticles = label(mask)
    props = regionprops(maskParticles)
    if len(props) == 1:
        return (fileName, props[0])
    else:
        return None

    
#This functions takes the path to a "/Mask" folder and applies the function "extractGeometryFromFile" to all the 
#"mask" files contained in it. It then returns a list of "skimage.measure._regionprops.RegionProperties" objects,
#one per file.
def processAWellFolder(folder : str = "") -> list:
    myFolder = Path(folder)
    filePatt = re.compile(r"^.*\.tif+$")
    fileList = [file.as_posix() for file in myFolder.iterdir() if file.is_file() and re.search(filePatt, file.as_posix())]
    fileList.sort()
    listOfParticles = []
    for file in fileList:
        result = extractGeometryFromFile(file)
        if result != None:
            listOfParticles.append(result)
        else:
            pass
    
    return listOfParticles
    

#This function takes the path of the folder containing all the folders of each well. It makes use of the function
#"processAWellFolder" to process all the files contained in each folder (remeber that you have to add "/Mask"
#to the path for "processAWellFolder"). It returns a dictionary where the "key" is a string corresponding to the
#“folder/well name“ and it cotains as unique value (for each key) a list of all the "RegionProperties" objects associated
#with that folder/well:
def processAllFolders(path : str = "") -> dict:
    mainPath = Path(path)
    folderList = [folder.as_posix() for folder in mainPath.iterdir() if folder.is_dir()]
    folderList.sort()
    dictionaryAllWells = dict()
    for folder in folderList:
        maskPath = folder + "/Masks"
        separator = re.compile(os.sep)
        mo = re.split(separator, folder)
        wellName = mo[-1]
        regionsList = processAWellFolder(maskPath)
        dictionaryAllWells[wellName] = regionsList
    
    return dictionaryAllWells


#It will be cool now to have a function that takes the dictionary, and build out of it a Pandas.DataFrame with the
#value of all the geometrical parameters I want to analyze and a label corresponding to the name of the well/folder 
#in which the original file was stored. The function return this DataFrame.
def createDataFrame(dictionary : dict = dict()):
    myDict = dictionary
    tempDict = {"Sample" : [], "Date" : [], "File_name" : [], "Area" : [], "Eccentricity" : [], "Major_Axis" : [],
                "Minor_Axis" : [], "Perimeter" : [], "Solidity" : [], "Plate_format" : [], "Treatment" : [],
                "Microwell_type" : [], "Microwell_diameter" : [], "Staining" : [], "Diameter_Of_Circle" : [],
                "Repetition" : []}
    
    for label in myDict:
        sepPatt = re.compile(r"_")
        mo = re.split(sepPatt, label)
        Date = mo[0]
        Plate_format = mo[1]
        Microwell_type = mo[2]
        Microwell_diameter = mo[3]
        Staining = mo[4]
        Treatment = mo[5]
        Repetition = mo[6]
        
        for elems in myDict[label]:
            tempDict["Sample"].append(label)
            tempDict["Date"].append(Date)
            tempDict["Plate_format"].append(Plate_format)
            tempDict["Microwell_type"].append(Microwell_type)
            tempDict["Microwell_diameter"].append(Microwell_diameter)
            tempDict["Staining"].append(Staining)
            tempDict["Treatment"].append(Treatment)
            tempDict["Repetition"].append(Repetition)
            
            #Extract from the tuple created by "extractGeometryFromFile" function the file name (= first element
            #of the tuple):
            tempDict["File_name"].append(elems[0])
            #Extract all the geometrical parameters that are part of the "RegionProperties" object, which is the
            #second element of the tuple:
            tempDict["Area"].append(elems[1].area)
            tempDict["Eccentricity"].append(elems[1].eccentricity)
            tempDict["Major_Axis"].append(elems[1].major_axis_length)
            tempDict["Minor_Axis"].append(elems[1].minor_axis_length)
            tempDict["Perimeter"].append(elems[1].perimeter)
            tempDict["Solidity"].append(elems[1].solidity)
            tempDict["Diameter_Of_Circle"].append(elems[1].equivalent_diameter)
    
    #Create the Pandas.DataFrame from "tempDict":
    myData = pd.DataFrame(tempDict)
    #Process the values in "myData" DataFrame:
    #Calculate the "Circularity" parameter ((4*pi*Area)/(Perimeter^2)):
    myData["Circularity"] = (4*math.pi*myData["Area"]) / (myData["Perimeter"]**2)
    #Calculate the "Proper_Roundness", the one defined by this formula ( (4*Area)/(pi*Major_Axis^2)  ):
    myData["Roundness"] = (4*myData["Area"]) / (math.pi*(myData["Major_Axis"]**2))
    
    #Convert all the values that are in pixel-dimensions to micrometer-dimensions:
    myData["Area"] = myData["Area"]*pixelArea
    myData["Major_Axis"] = myData["Major_Axis"]*pixelLength
    myData["Minor_Axis"] = myData["Minor_Axis"]*pixelLength
    myData["Perimeter"] = myData["Perimeter"]*pixelLength
    myData["Diameter_Of_Circle"] = myData["Diameter_Of_Circle"]*pixelLength
    
    myData = myData[["Sample", "File_name", "Date", "Plate_format", "Microwell_type", "Microwell_diameter", "Staining", "Treatment",
                     "Repetition", "Area", "Perimeter", "Diameter_Of_Circle", "Roundness", "Major_Axis", "Minor_Axis", "Circularity", "Eccentricity", 
                     "Solidity"]]
    return myData


    
#EXECUTE THE CODE:
#Create the DataFrame:
myDictionary = processAllFolders(path)
myData = createDataFrame(myDictionary)
print(myData)



#This functions takes a proper Pandas.DataFrame and represent the data in it in form of Seaborn violinplots:
def violinPlotGeometry(df, xName : str = "Sample", yName : str = "", hueStr : str = None, fileName : str = "ViolinPlot"):
    fig, ax = plt.subplots(1)
    fig.suptitle(yName)
    sns.violinplot(x = xName, y = yName, data = df, ax = ax, hue = hueStr)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = "right")
    fig.set_size_inches(12, 6)
    plt.show();
    fig.savefig(path + os.sep + fileName + ".pdf", dpi = 300, format = "pdf", bbox_inches = "tight")
    
    return fig, ax

#This functions takes a proper Pandas.DataFrame and represent the data in it in form of Seaborn barplots
#(average as height of bars and standard deviation reported as a line):
def barPlotGeometry(df, xName : str = "Sample", yName : str = "", hueStr : str = None, fileName : str = "BarPlot"):
    fig, ax = plt.subplots(1)
    fig.suptitle(yName)
    sns.barplot(x = xName, y = yName, data = df, ci = "sd", hue = hueStr)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = "right")
    ax.set_ylabel(ax.get_ylabel() + "  (Mean and SD)")
    fig.set_size_inches(12, 6)
    plt.show();
    fig.savefig(path + os.sep + fileName + ".pdf", dpi = 300, format = "pdf", bbox_inches = "tight")
    
    return fig, ax

#Strip plot function:
def stripPlotGeometry(df, xName : str = "Sample", yName : str = "", hueStr : str = None, fileName : str = "StripPlot"):
    fig, ax = plt.subplots(1)
    fig.suptitle(yName)
    sns.stripplot(x = xName, y = yName, hue = hueStr, data = df, jitter = True, ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = "right")
    fig.set_size_inches(12, 6)
    plt.show();
    fig.savefig(path + os.sep + fileName + ".pdf", dpi = 300, format = "pdf", bbox_inches = "tight")
    
    return fig, ax

#Swarm plot function:
def swarmPlotGeometry(df, xName : str = "Sample", yName : str = "", hueStr : str = None, fileName : str = "SwarmPlot"):
    fig, ax = plt.subplots(1)
    fig.suptitle(yName)
    sns.swarmplot(x = xName, y = yName, hue = hueStr, data = df, ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = "right")
    fig.set_size_inches(12, 6)
    plt.show();
    fig.savefig(path + os.sep + fileName + ".pdf", dpi = 300, format = "pdf", bbox_inches = "tight")
    
    return fig, ax

def boxPlotGeometry(df, xName : str = "Sample", yName : str = "", hueStr : str = None, fileName : str = "BoxPlot"):
    fig, ax = plt.subplots(1)
    fig.suptitle(yName)
    sns.boxplot(x = xName, y = yName, hue = hueStr, data = df, ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = "right")
    fig.set_size_inches(12, 6)
    plt.show();
    fig.savefig(path + os.sep + fileName + ".pdf", dpi = 300, format = "pdf", bbox_inches = "tight")
    
    return fig, ax


#CREATE THE PLOTS:
#Violin plot:
violinPlotGeometry(myData, yName = "Area")
#Various Bar plots:
barPlotGeometry(myData, yName = "Area")
barPlotGeometry(myData, yName = "Area", hueStr = "Plate_format", fileName = "BarPlotByPlateFormat")
barPlotGeometry(myData, xName = "Microwell_diameter", yName = "Area", hueStr = "Plate_format", fileName = "BarPlotByPlateFormatAndMicrowellDiameter")
#Strip plot:
stripPlotGeometry(myData, yName = "Area")
#Swarm plot:
swarmPlotGeometry(myData, yName = "Area")
#Box plot:
boxPlotGeometry(myData, yName = "Area")





#Here you can calculate various parameters from "myData" DataFrame:

#Average Area per Sample:
sampleGroup = myData.groupby("Sample")
sampleGroup.agg(["mean", "std"])
summary = sampleGroup.agg(["mean", "std"])

myData.to_csv(path + os.sep + "Data.csv", index = False)
summary.to_csv(path + os.sep + "DataSummary.csv", index = True)





