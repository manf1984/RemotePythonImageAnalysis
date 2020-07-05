'''
Created on 5 Jul 2020

@author: Andrea Manfrin
'''

import os
import shutil
import re
from pathlib import Path
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from skimage import morphology as skm
import scipy.ndimage.morphology as scm
from skimage.external.tifffile import TiffFile
from skimage.external.tifffile import TiffWriter
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


#Test function for Buttons:
def OK():
    print("OK!!")


#This class, when instantiated, requires (as arguments of .__init__) the path of a "mask_overlay" file (in str format) 
#and a valid "tkinter.Canvas" object (the one inside whoch all the microwells images will be placed).
#It will create a series of useful microwell-associated attributes:
class Microwell(object):
    #Class variables - folders related to the well to which the file/microwell belongs:
    wellFolder = None
    overlayFolder = None
    allChannelsFolder = None
    maskFolder = None
    #Class variables - current vertical position on the Canvas in which insert the inner-canvas
    #containing the image of the current microwell. This values is updated for the class everytime
    #a new Microwell object is created:
    verticalEntryPosition = 0
    index = 0
    
    #These are the instance methods that will be executed by the Buttons at the bottom of each Mask_Overlay image.
    #They will take care specifically of that file/microwell (= specific for each Microwell instance)!
    def secondSegmentation(self): #NOT DEFINED YET
        print("second segm", self.allChannelsFile)
    
    def manualSegmentation(self):
        #THIS METHOD CREATES AN INSTANCE OF THE CLASS "ManualSegWindow" CONTAINING THE IMAGE OF THE
        #MICROWELL YOU WANT TO SEGMENT (= the one from which you press the Button "Manual Segmentation").
        #It also returns to the constructor of "ManualSegWindow" a lot of varaiables associated with the
        #specific microwell:
        ManualSegWindow(self.allChannelsFile, self.overlayFile , self.maskFile, self.microwellIndex, self.innerCanvas)
    
    def remove(self): #NOT DEFINED YET
        print("remove", self.allChannelsFile)
    
    def restore(self): #NOT DEFINED YET
        print("restore", self.allChannelsFile)
    
    
    #This is the __init__ method (= constructor) called when a new Microwell instance is created.
    #The "canvas" argument is a tkinter.Canvas object to which the widgets associated with each Microwell instance
    #will be added one after the other.
    #This is possible because the Microwell class has a Class Variable called "verticalEntryPosition" that keeps
    #track of the point of the "canvas" where the last widgets have been added and it starts from that point to insert
    #the next ones!
    def __init__(self, overlayFilePath : str, canvas : tk.Canvas = None):
        self.__class__.wellFolder = "" #You need to reset this Class Variable every time you create a new Microwell object
        patt = re.compile(os.sep)
        mo = re.split(patt, overlayFilePath)
        #Instantiate Class variables:
        for folder in mo[1:-2]:
            piece = os.sep + folder
            self.__class__.wellFolder += piece
        
        self.__class__.overlayFolder = self.__class__.wellFolder + os.sep + "Mask_Overlay"
        self.__class__.allChannelsFolder = self.__class__.wellFolder + os.sep + "All_Channels_And_Mask"
        self.__class__.maskFolder = self.__class__.wellFolder + os.sep + "Masks"
        
        #Instantiate Instance variables (associated with a specific series of microwell files).
        #Paths to the microwell-associated files: 
        self.overlayFile = self.__class__.overlayFolder + os.sep + mo[-1]
        self.allChannelsFile = self.__class__.allChannelsFolder + os.sep + mo[-1]
        self.maskFile = self.__class__.maskFolder + os.sep + mo[-1]
        #Get the microwell name from the file's name:
        name = mo[-1]
        pattTif = re.compile(r"\.")
        moFile = re.split(pattTif, name)
        self.fileName = moFile[0]
        
        
        #Create a PhotoImage object from the "mask_overlay" file asscoiated with this microwell:
        self.image = ImageTk.PhotoImage(Image.open(self.overlayFile))
        
        #Create a LabelFrame with the name of the file/microwell on top of each Canvas (innerCanvas) containig the
        #image of the microwell; this LabelFrame is also contained in the general "canvas" Canvas:
        self.fileNameLabel = tk.LabelFrame(canvas, width = 300, height = 30, text = "Microwell " + self.fileName, fg = "black", font = ("Arial", 30, "bold"), bd = 0)
        canvas.create_window((self.image.width()/2 - 90, self.__class__.verticalEntryPosition), window = self.fileNameLabel, anchor = tk.N + tk.W)
        #Update the Class Variable "verticalEntryPosition", so that the next object is inserted
        #below the previous one:
        self.__class__.verticalEntryPosition = self.__class__.verticalEntryPosition + 31
        
        #Set the width of the "canvas" equal to the width of the image
        #(all microwells files must have the same width!!):
        canvas.config(width = self.image.width())
        
        #Create the innerCanvas that will contain "self.image":
        self.innerCanvas = tk.Canvas(canvas, width = self.image.width(), height = self.image.height())
        self.innerCanvas.create_image((0, 0), anchor = "nw", image = self.image)  #Create the image inside the innerCanvas.
        #CREATE A WINDOW INSIDE THE OUTER CANVAS (canvas) TO DISPLAY THE CONTENT OF THE innerCanvas!!!!!!
        canvas.create_window((0, self.__class__.verticalEntryPosition), window = self.innerCanvas, anchor="nw")  #YOU HAVE TO SET THE Y-POSITION OF EACH WINDOW CONTAINING EACH INNER CANVAS INSIDE THE OUTER CANVAS!!
        
        #Update the Class Variable "verticalEntryPosition", so that the next object is inserted
        #below the previous one:
        self.__class__.verticalEntryPosition = self.__class__.verticalEntryPosition + self.image.height() + 1
        
        #FRAME OF SEPARATION BETWEEN TWO PICTURES!
        self.separationFrame = tk.Frame(canvas, bg = "gray", height = 40, width = self.image.width())
        self.separationFrame.grid_propagate(False) #This will avoid that when a Button is inserted in the Frame it will change the Frame geometry!!!
        #Add the 4 Buttons to the "separationFrame":
        self.buttonSecondSeg = tk.Button(self.separationFrame, text = "Second Segmentation", command = self.secondSegmentation, bd = 0, highlightthickness = 0)
        self.buttonManualSeg = tk.Button(self.separationFrame, text = "Manual Segmentation", command = self.manualSegmentation, bd = 0, highlightthickness = 0)
        self.buttonRemove = tk.Button(self.separationFrame, text = "Remove", command = self.remove, bd = 0, highlightthickness = 0)
        self.buttonRestore = tk.Button(self.separationFrame, text = "Restore", command = self.restore, bd = 0, highlightthickness = 0)
        self.buttonSecondSeg.grid(row = 0, column = 0, padx = 90, pady = 7)
        self.buttonManualSeg.grid(row = 0, column = 1, padx = 90, pady = 7)
        self.buttonRemove.grid(row = 0, column = 2, padx = 90, pady = 7)
        self.buttonRestore.grid(row = 0, column = 3, padx = 90, pady = 7)
        
        #CREATE A WINDOW INSIDE THE OUTER CANVAS (canvas) TO DISPLAY THE separationFrame!
        canvas.create_window((0, self.__class__.verticalEntryPosition), window = self.separationFrame, anchor = "nw")
        
        #Update again the Class Variable "verticalEntryPosition", so that the next object is inserted
        #below the previous one:
        self.__class__.verticalEntryPosition = self.__class__.verticalEntryPosition + 60
        
        #Record the index of the microwellat every instantion. Take it from the the Class Variable "index" and
        #store it in the Instance Variable "microwellIndex":
        self.microwellIndex = self.__class__.index
        self.__class__.index += 1


class InspectionWindow(object):
    #These are the fun tions associated witn the Buttons "self.buttonConfirm" and "self.buttonCancel".
    #
    #Function for "self.buttonConfirm". It will copy the files contained in the "mod_files" folders sub-folders
    #(by pressing the "OK" button in the "SegmentedWindow" opened by a "ManualSegWindow" object), in the
    #corresponding folders present in the main folder (the "well folder" if you want...). Then it will delete the
    #"mod_files" folder with all its content and close the window associated with the "InspectionWindow" object.
    def confirmChanges(self):
        try:
            #Get the file paths for each file of each sub-folder of the "mod_files" folder.
            #
            #Here I will get the file paths of the "All_Channels_And_Mask" folder:
            allChannelsFolderString = self.mainFolderPath + os.sep + "mod_files" + os.sep + "All_Channels_And_Mask"
            allChannelsFolderPath = Path(allChannelsFolderString)
            newAllChannelsFileList = [file.as_posix() for file in allChannelsFolderPath.iterdir() if file.is_file()]
            newAllChannelsFileList.sort()
            #List that will contain the file paths to substitute/overwrite:
            oldAllChannelsFileList = []
            #This compiled Regular Expression can be used also for the other two sub-folders lists!
            mod_filesPatt = re.compile(r"/mod_files")
            #Add the elements to "oldAllChannelsFileList":
            for filePath in newAllChannelsFileList:
                mo = re.split(mod_filesPatt, filePath)
                modPath = mo[0] + mo[1]
                oldAllChannelsFileList.append(modPath)

            #Here I will get the file paths of the "Masks" folder:
            masksString = self.mainFolderPath + os.sep + "mod_files" + os.sep + "Masks"
            masksPath = Path(masksString)
            newMasksFileList = [file.as_posix() for file in masksPath.iterdir() if file.is_file()]
            newMasksFileList.sort()
            #List that will contain the file paths to substitute/overwrite:
            oldMasksFileList = []
            #Add the elements to "oldMasksFileList":
            for filePath in newMasksFileList:
                mo = re.split(mod_filesPatt, filePath)
                modPath = mo[0] + mo[1]
                oldMasksFileList.append(modPath)

            #Here I will get the file paths of the "Mask_Overlay" folder:
            maskOverlayString = self.mainFolderPath + os.sep + "mod_files" + os.sep + "Mask_Overlay"
            maskOverlayPath = Path(maskOverlayString)
            #Select only the ".tif" files from the "mod_files/Mask_Overlay" folder (not the ".png"):
            newMaskOverlayFileList = [file.as_posix() for file in maskOverlayPath.iterdir() if file.is_file() and re.search(re.compile(r".tif+$"), file.as_posix())]
            newMaskOverlayFileList.sort()
            #List that will contain the file paths to substitute/overwrite:
            oldMaskOverlayFileList = []
            #Add the elements to "oldMaskOverlayFileList":
            for filePath in newMaskOverlayFileList:
                mo = re.split(mod_filesPatt, filePath)
                modPath = mo[0] + mo[1]
                oldMaskOverlayFileList.append(modPath)

            #Now overwrite the files using the new and old fileLists created above.
            #The files are ordered in the same way bewteen the corresponding list, thus I can use "for loops"
            #based on indeces.
            #
            #Copy the files of the folder "/mod_files/All_Channels_And_Mask" to "All_Channels_And_Mask":
            for i in range(0, len(newAllChannelsFileList), 1):
                shutil.copy(newAllChannelsFileList[i], oldAllChannelsFileList[i])
            #Copy the files of the folder "/mod_files/Masks" to "Masks":
            for i in range(0, len(newMasksFileList), 1):
                shutil.copy(newMasksFileList[i], oldMasksFileList[i])
            #Copy the files of the folder "/mod_files/Mask_Overlay" to "Mask-Overlay":
            for i in range(0, len(newMaskOverlayFileList), 1):
                shutil.copy(newMaskOverlayFileList[i], oldMaskOverlayFileList[i])

            #Delete the "/mod_files" folder with all its content:
            pathToRemove = self.mainFolderPath + os.sep + "mod_files"
            shutil.rmtree(pathToRemove)

            #Close the window associated with this "InspectionWindow" object:
            self.root.destroy()
            self.root.quit()
        except FileNotFoundError:
            print("You have not modified any image/file yet. There is nothing to Confirm.")
        
#     #Function for "self.buttonCancel". 
#     def cancelAction(self):
#         #Delete the "/mod_files" folder with all its content:
#         pathToRemove = self.mainFolderPath + os.sep + "mod_files"
#         try:
#             shutil.rmtree(pathToRemove)
#         except FileNotFoundError:
#             pass
        
#         #Close the window associated with this "InspectionWindow" object:
#         self.root.destroy()
#         self.root.quit()
    
    
    #Function for "self.buttonCancel". 
    def cancelAction(self):
        #The dialog window asking if you are sure you want to quit will be created only if
        #the "/mod_files" folder has been created = if some files/images have been modified and not
        #yet "confirmed":
        pathToCheck = self.mainFolderPath + os.sep + "mod_files"
        if os.path.isdir(pathToCheck):
            self.confirmWindow = tk.Toplevel()
            self.confirmWindow.title("Are you sure?")
            self.confirmFrame = tk.Frame(self.confirmWindow, width = 300, height = 150)
            self.confirmFrame.grid(row = 0, column = 0)

            def yesClose():
                #Delete the "/mod_files" folder with all its content:
                pathToRemove = self.mainFolderPath + os.sep + "mod_files"
                try:
                    shutil.rmtree(pathToRemove)
                except FileNotFoundError:
                    pass

                #Close the window associated with this "InspectionWindow" object:
                self.root.destroy()
                self.root.quit()

                #Close this dialog window:
                self.confirmWindow.destroy()
                self.confirmWindow.quit()


            def noClose():
                #Close this dialog window:
                self.confirmWindow.destroy()
                self.confirmWindow.quit()

            self.labelText = tk.Label(self.confirmFrame,
                                      text = "Do you want to close this window?\n If you close this window before pressing the \"Confirm\" button\n all the modifications you applied will be lost!",
                                      pady = 20, anchor = tk.CENTER)
            self.labelText.grid(row = 0, column = 0, columnspan = 2, sticky = tk.E + tk.W)
            self.yesButton = tk.Button(self.confirmFrame, text = "Yes", command = yesClose, bd = 0)
            self.noButton = tk.Button(self.confirmFrame, text = "No", command = noClose, bd = 0)
            self.yesButton.grid(row = 1, column = 0, padx = (50, 20), pady = (50, 50))
            self.noButton.grid(row = 1, column = 1, padx = (20, 50), pady = (50, 50))

            self.confirmWindow.mainloop()
        else: #No need to ask. There is nothing to lose!
            #Close the window associated with this "InspectionWindow" object:
            self.root.destroy()
            self.root.quit()
    
    #When you instantiate this class you create an instance which gives rise to the window with all
    #the images of the overlay_mask files and all the widgets associated. This is a "second level" window
    #a "tk.Toplevel" object.
    #To create an instance it is required to give to the constructor a full path (in str form) to one of the
    #wells. The main one; the code in the __init__ method will take care to rebuild the path to the associated 
    #"Mask_Overlay" folder.
    def __init__(self, path : str = ""):
        #Create the path to the associated "Mask_Overlay" folder starting from the "path" argument of the
        #"__init__" method:
        self.mainFolderPath = path
        self.path = path + os.sep + "Mask_Overlay"
        
        #Get all the file paths and place them in a list:
        pathPath = Path(self.path)
        pattFile = re.compile(r"^.*\.tif+$")
        pathList = [file.as_posix() for file in pathPath.iterdir() if file.is_file() and re.match(pattFile, file.as_posix())]
        pathList.sort()
        
        #Create the "second level" window, whixh is a tk.Toplevel object that I call "root":
        self.root = tk.Toplevel()
        self.root.title("Microwell Inspection Window")
        
        #Create the Canvas and the Scrollbar inside the Tk object "root" and associate them in terms of commands
        #(EventListener and Event):
        self.canvas = tk.Canvas(self.root, height = 800)
        self.scrollbar = tk.Scrollbar(self.root, orient = "vertical", command = self.canvas.yview)
        self.canvas.config(yscrollcommand = self.scrollbar.set)
        
        #Create a list with all the Microwell objects. Each of them will modify the "canvas" object created, and will add
        #to it its components (innerCanvas and separationFrame):
        self.microwellList = [Microwell(mwell, self.canvas) for mwell in pathList]
        
        #Adjust the "scrollregion" of "canvas" to the full area occupied by all the widgets contained at the end in "canvas":
        self.canvas.config(scrollregion = self.canvas.bbox("all"))
        
        #Create the "endFrame" that will stay at the bottom of "canvas" and "scrollbar":
        self.endFrame = tk.Frame(self.root, bg = "gray", height = 50)
        self.endFrame.grid_propagate(False) #This will avoid that when a Button is inserted in the Frame it will change the Frame geometry!!!
        
        #Create the Buttons contained in "endFrame":
        self.buttonConfirm = tk.Button(self.endFrame, text = "Confirm", font = ("Helvetica", 30), command = self.confirmChanges, bd = 0, highlightthickness = 0)
        self.buttonCancel = tk.Button(self.endFrame, text = "Cancel", font = ("Helvetica", 30), command = self.cancelAction, bd = 0, highlightthickness = 0)
        self.buttonConfirm.grid(row = 0, column = 0, padx = (450, 30), pady = 5)
        self.buttonCancel.grid(row = 0, column = 1, padx = (30, 450), pady = 5)

        #Pack (using .grid) all the widgets inside "root" ("canvas, "scrollbar", "endFrame"):
        self.canvas.grid(row = 0, column = 0)
        self.scrollbar.grid(row = 0, column = 1, sticky = tk.N + tk.S)
        self.endFrame.grid(row = 1, columnspan = 2, sticky = tk.E + tk.W)

        #Bind an Event Listener (in this case using ".protocol") to the Toplevel object "self.root" for the "Window Closure".
        #This will trigger the same method triggered by the "self.buttonCancel" Button that will delete
        #eventually the "/mod_files" folder if present and then terminate the window:
        self.root.protocol("WM_DELETE_WINDOW", self.cancelAction)
        
        
        self.root.mainloop()

########I WANT TO CHANGE THIS CLASS SO THAT IT IS POSSIBLE TO START FROM THE AUTOMATICALLY GENRATED MASK AND##########
########ADD OR REMOVE AREAS FROM IT. THIS REQUIRES A DEPP RETHINKING OF THIS CLASS.###################################
class ManualSegWindow(object):
    #Instance methods that defines the Actions performed when Event Listeners are triggered: 
    def getCoordinates(self, event): #This method should be already ok!
        #List of points (x,y) or (cols, rows) manually selected drawing the segmentation trace on the image:
        self.coordinatesList.append((event.x, event.y))
        
        #Code to draw the segmentation trace on the image and update the image inside the Label at every point
        #added to the segmentation trace:
        self.imageRGB[event.y, event.x, :] = (247, 202, 24, 255)
        self.photoImageRGB = ImageTk.PhotoImage(Image.fromarray(self.imageRGB))
        self.label.config(image = self.photoImageRGB)
        
    #This method is executed when the mouse Button1 is released. It will remove eventually duplicated points present
    #in "self.coordinatesList":
    def cleanCoordinates(self, event):
        self.coordinatesList = list(set(self.coordinatesList)) #Passing through a "set" object duplicated points are removed!
        
    #This code is executed when the Button "self.acceptButton" is pressed. It will create another Toplevel window 
    #showing the segmented region.
    def useCoordinates(self):
        colList = [c[0] for c in self.coordinatesList] #The ordered list of x-coordinates for each point. These will correspond to the column number!
        rowList = [c[1] for c in self.coordinatesList] #The ordered list of y-coordinates for each point. These will correspond to the row number!
        
        self.selection = np.zeros((self.image.shape[0], self.image.shape[1])) #create an empty "ndarray"
        self.selection = self.selection.astype(np.uint8)
        self.selection[rowList, colList] = 1  #Transpose the "drawn" points in the empty array. They will acquire a value of 1.
        #Process a bit the points with a square-dilation of 9 pixels (= 4 around each point in  every direction),
        #close the perimeter of the selection and fill its inside. The selected area is now ready:
        self.selection = skm.binary_dilation(self.selection, selem = skm.square(9))
        self.selection = skm.binary_closing(self.selection)
        self.selection = scm.binary_fill_holes(self.selection)
        #Mask the original image ("self.image") using "self.selection":
        img = self.image * self.selection
        img = ImageTk.PhotoImage(Image.fromarray(img))
        
        #Create a new Toplevel window and Label to visualize the segemented object ("img"):
        self.segmentedWindow = tk.Toplevel()
        self.segmentedWindow.title("Segmented Object")
        self.segmentedLabel = tk.Label(self.segmentedWindow, image = img, bd = 0, padx = 0, pady = 0)
        self.segmentedLabel.grid(row = 0, column = 0)
        
        #Functions used by the two Buttons (instatiated below) "self.acceptSegmentedButton" and
        #"self.cancelSegmentedButton".
        #
        #This is the function executed when the "self.acceptSegmentedButton" is pressed.
        #It will create a temporary directory "mod_files" inside the folder of the "well" and inside it 3 sub-folders:
        #"overlay" "allChannels" and "mask".
        #Inside each of these folders it will create a (respectively) a new "overlay file", a new "allChannels file"
        #and a new "mask file" (these files will have the segmentation mask updated to the one manually drawn).
        #It will then delete the image from "self.innercanvas" and insert the new "overlay file" image!!
        #(The three files created will substitute the original ones only and only when you press the "Confirm" Button
        #in the "InspectionWindow". After that also the "mod_files" folder with all the files will be deleted)
        def acceptSegmentation():
            #Extract pieces of path for all the diffrent file path and for the general "mod_files" folder
            pattSep = re.compile(r"/")
            moOverlay = re.split(pattSep, self.overlayFile)
            moAllChannels = re.split(pattSep, self.allChannelsFile)
            moMask = re.split(pattSep, self.maskFile)
            
            #Create the path for the "mod_files" folder
            self.mod_filesFolder = ""
            for piece in moOverlay[1:-2]:
                self.mod_filesFolder = self.mod_filesFolder + os.sep + piece
            self.mod_filesFolder = self.mod_filesFolder + os.sep + "mod_files"

            #Create the paths for the three sub-folders inside "mod_filesFolder" and for the files that will be saved
            #each of these sub-folders:
            moOverlay.insert(-2, "mod_files")
            self.newOverlayFolder = ""
            self.newOverlayFile = ""
            for piece in moOverlay[1:-1]:
                self.newOverlayFolder = self.newOverlayFolder + os.sep + piece
            for piece in moOverlay[1:]:
                self.newOverlayFile = self.newOverlayFile + os.sep + piece
            
            moAllChannels.insert(-2, "mod_files")
            self.newAllChannelsFolder = ""
            self.newAllChannelsFile = ""
            for piece in moAllChannels[1:-1]:
                self.newAllChannelsFolder = self.newAllChannelsFolder + os.sep + piece
            for piece in moAllChannels[1:]:
                self.newAllChannelsFile = self.newAllChannelsFile + os.sep + piece
            
            moMask.insert(-2, "mod_files")
            self.newMaskFolder = ""
            self.newMaskFile = ""
            for piece in moMask[1:-1]:
                self.newMaskFolder = self.newMaskFolder + os.sep + piece
            for piece in moMask[1:]:
                self.newMaskFile = self.newMaskFile + os.sep + piece 
            
            #Creation of the actual "mod_filesFolder" and of all the three sub-folders contained in it:
            #The creation of each folder is inside a "try" block. If the folder already exists a "FileExistsError"
            #will be raised. This is catched by the "except" block that executes a "pass" statement (= does nothing).
            #The result in case of exception is that nothing happens (no overwriting of any folder) and the
            #"FileExistsError" does not block the execution of the program:
            try:
                os.mkdir(self.mod_filesFolder)
            except FileExistsError:
                pass
            
            try:
                os.mkdir(self.newOverlayFolder)
            except FileExistsError:
                pass
            
            try:
                os.mkdir(self.newAllChannelsFolder)
            except FileExistsError:
                pass
            
            try:
                os.mkdir(self.newMaskFolder)
            except FileExistsError:
                pass
            
            #Create the content of the new files and save the files in the corresponding sub-folder:
            #"All_Channels_And_Mask" file:
            mod_filesImageAllCh = None
            with TiffFile(self.allChannelsFile) as tif:
                mod_filesImageAllCh = tif.asarray()
                tif.close()
            if mod_filesImageAllCh.ndim == 4: #In case the images were not Z-Projected (so they still have 4 axis) and "self.selection" must be inserted in each Z-Stack!
                mod_filesImageAllCh[:, 4, :, :] = self.selection
            else: #In case the images were Z-Projected, so they have 3 axis!! (The most common case)
                mod_filesImageAllCh[4, :, :] = self.selection
                
            with TiffWriter(self.newAllChannelsFile, bigtiff = False, imagej = True) as writer:
                writer.save(mod_filesImageAllCh)
                writer.close()
            
            #"Masks" file:
            with TiffWriter(self.newMaskFile, bigtiff = False, imagej = True) as writer:
                writer.save(self.selection.astype(np.uint8))
                writer.close()
            
            #"Mask_Overlay":
            #Take the BF from "mod_filesImageAllCh". If not Z-Projected make the "mean" projection:
            BFImage = mod_filesImageAllCh
            if BFImage.ndim == 4: #Not Z-Projected
                BFImage = np.around(BFImage.mean(axis = 0)).astype(np.uint8)
                BFImage = BFImage[0, :, :]
            else: #Z-Projected
                BFImage = BFImage[0, :, :]
            
            #Create the overlay of BF and Mask using "imshow" method of Matplotlib Axes.
            #Then save the ".tif" file using the Matplotlib Figure.savefig method (in the "mod_files/Mask_Overlay" folder):
            fig, ax = plt.subplots(1)
            ax.imshow(BFImage, cmap = "gray")
            ax.imshow(self.selection.astype(np.uint8)*250, cmap = "inferno", alpha = 0.2)
            ax.axis("off")
            fig.savefig(self.newOverlayFile, format = "tif", dpi = 200)
            
            #Code to change the image on the microwell-specific "innerCanvas" of "InspectionWindow" with the new
            #"mask_overlay" image:
            #(Add also something to show that this image was re-segmented)
            ax.text(x = 0.02, y = 0.97, s = "Manually Re-segmented", fontsize = 6, color = "yellow", transform=ax.transAxes)
            #Now I have to rebuild a PhotoImage with this new version (with label) of the "mask_overlay" and save it on
            #a file. The saving on a file is the ONLY way to merge all the ImageAxes together in a single file and
            #give this merged RGB image to the Image constructor (and then to PhotoImage). I will use a ".png" this time. 
            #Let's create the file path for this new temporary file/image:
            removeTif = re.compile(r".tif+$")
            moNoTif = re.split(removeTif, self.newOverlayFile)
            self.overlayWithLabelPath = moNoTif[0] + "_withLabel.png"
            fig.savefig(self.overlayWithLabelPath, format = "png", dpi = 200)
            
            #Create the PhotoImage object form the .png file:
            self.imgWithLabel = ImageTk.PhotoImage(Image.open(self.overlayWithLabelPath))
            #Delete the previous image from "innerCanvas" and place the new one: 
            self.innerCanvas.delete("all")
            #Load on innerCanvas the new image:
            self.innerCanvas.create_image((0, 0), anchor = "nw", image = self.imgWithLabel)
            
            #Close the Figure instance "fig", otherwise these objects will accumualte in the memory slowering
            #down the program:
            plt.close(fig)
            
            #Terminate the (one-level upstream) window associated with "ManualSegWindow" (= terminate
            #"self.manualSegWin"):
            self.manualSegWin.destroy()
            self.manualSegWin.quit()
            
            #Terminate the "self.segmentedWindow":
            self.segmentedWindow.destroy()
            self.segmentedWindow.quit()
            
            
        
        #This is the function executed when the "self.cancelSegmentedButton" is pressed:
        def refuseSegmentation():
            #Terminate the "self.segmentedWindow":
            self.segmentedWindow.destroy()
            self.segmentedWindow.quit()
            
            
        #Create a Frame with two Buttons:
        self.frameSegmented = tk.Frame(self.segmentedWindow, height = 60, bg = "gray")
        self.acceptSegmentedButton = tk.Button(self.frameSegmented, text = "OK", command = acceptSegmentation, bd = 0, highlightthickness = 0)
        self.cancelSegmentedButton = tk.Button(self.frameSegmented, text = "Cancel", command = refuseSegmentation, bd = 0, highlightthickness = 0)
        self.frameSegmented.grid(row = 1, column = 0, sticky = tk.W + tk.E)
        self.acceptSegmentedButton.grid(row = 0, column = 0, padx = (300, 50), pady = 7)
        self.cancelSegmentedButton.grid(row = 0, column = 1, padx = (50, 300), pady = 7)
        
        self.segmentedWindow.mainloop()
        

    #This code is executed when the Button "self.deleteButton" is pressed. It will delete all the points (= tuples)
    #from "self.coordinatesList" and cancel the segmentation trace from the image contained in the window:
    def deleteCoordinates(self):
        self.coordinatesList = []
        
        self.imageRGB = self.image.copy()
        grayMap = get_cmap("gray")
        self.imageRGB = grayMap(self.imageRGB)
        self.imageRGB = np.around(self.imageRGB * 255).astype(np.uint8)
        self.photoImageRGB = ImageTk.PhotoImage(Image.fromarray(self.imageRGB))
        self.label.config(image = self.photoImageRGB)
    
    #This function is executed when "self.cancelButton" is pressed.
    #It will just close the "self.manualSegWin":
    def cancelButtonAction(self):
        self.manualSegWin.destroy()
        self.manualSegWin.quit()
    
    
    #The instantiation of the objects of this class is done by the methods associated with the Buttons
    #related with each "mask_overlay" image in the "InspectionWindow" object.
    #Those methods create the object using the path "allChannelsFile" of that Microwell object as argument
    #for the constructor (= "__init__" method).
    def __init__(self, pathAllChannels : str = "", pathOverlay : str = "", pathMask : str = "", microwellIndex : int = None, innerCanvas = None):
        #These are all the paths (in str) to all the files related to the microwell "opened" in this "ManualSegWindow":
        self.allChannelsFile = pathAllChannels
        self.overlayFile = pathOverlay
        self.maskFile = pathMask
        #This is the index/position of the microwell in the Canvas of the "InspectionWindow":
        self.microwellIndex = microwellIndex
        #This is the reference to the "innerCanvas" (present in the Canvas of the "InspectionWindow") that
        #contains the image of this specific microwell (it is required to modify/update the image
        #in "InspectionWindow"):
        self.innerCanvas = innerCanvas

        #Here I store the BF channel picture contained in the .tif file in "self.image" (as an unsigned 8bit ndarray).
        #In case the Z-stycks are not yet projected I will first do that. I that case I will use the "mean" of the
        #Z-Stacks. The BF channel must be the first channel (= 0).
        self.image = None
        
        with TiffFile(self.allChannelsFile) as tif:
            self.image = tif.asarray()
        if self.image.ndim == 4:
            self.image = round(self.image.mean(axis = 0))
            self.image = self.image.astype(np.uint8)
            self.image = self.image[0, :, :]
        else:
            self.image = self.image[0, :, :]
        
        
        #This is the list holding all the coordinates for the segmentation:
        self.coordinatesList = []
        #Here I will load in "self.coordinatesList" all the coordinates of the mask already calculated by
        #the automatic-segmentation:
        #################################################################################################
        #################################################################################################
        #############FROM HERE!!!!!!!!!!!!###############################################################
        #################################################################################################
        #################################################################################################
        
        #This is a copy of "self.image" that is used to update the image adding to it the traces of the
        #manual segmentation and is converted in RGB according to the "gray" Matplotlib ColorMap:
        self.imageRGB = self.image.copy()
        grayMap = get_cmap("gray")
        self.imageRGB = grayMap(self.imageRGB)
        #This is the final array of the original "self.image" in RGB format (RGB colors extracted from the "gray"
        #ColorMap):
        self.imageRGB = np.around(self.imageRGB * 255).astype(np.uint8)
        #This is the PhotoImage object derived from the RGB array version of the self.image array(= self.imageRGB):
        self.photoImageRGB = ImageTk.PhotoImage(Image.fromarray(self.imageRGB))
        
        #Create the "third level" window (another "tkinter.Toplevel" object):
        self.manualSegWin = tk.Toplevel()
        self.manualSegWin.title("Manual Segmentation Window")
        
        #Embed the PhotoImage "self.photoImageRGB" in a "tk.Label" which is contained in the Toplevel 
        #window "self.manualSegWin":
        self.label = tk.Label(self.manualSegWin, image = self.photoImageRGB, bd = 0, padx = 0, pady = 0)
        self.label.grid(row = 0, column = 0)
        
        #Associate Event Listeners and Actions (methods) to "self.label" where the "PIL Image" derived from
        #"self.image" is embedded:
        self.label.bind("<B1-Motion>", self.getCoordinates)
        self.label.bind("<ButtonRelease-1>", self.cleanCoordinates)
        
        #Create the Frame and Buttons for this new window:
        self.buttonFrame = tk.Frame(self.manualSegWin, height = 60, bg = "gray")
        self.buttonFrame.grid(row = 1, column = 0, sticky = tk.W + tk.E)
        self.acceptButton = tk.Button(self.buttonFrame, text = "Segment", command = self.useCoordinates, bd = 0, highlightthickness = 0)
        self.deleteButton = tk.Button(self.buttonFrame, text = "Delete Selection", command = self.deleteCoordinates, bd = 0, highlightthickness = 0)
        self.cancelButton = tk.Button(self.buttonFrame, text = "Cancel", command = self.cancelButtonAction, bd = 0, highlightthickness = 0)
        self.acceptButton.grid(row = 1, column = 0, padx = (300, 0), pady = 5)
        self.deleteButton.grid(row = 1, column = 1, padx = (50, 50), pady = 5)
        self.cancelButton.grid(row = 1, column = 2, padx = (0, 300), pady = 5)
        
        #Run the Toplevel window "self.manualSegWin":
        self.manualSegWin.mainloop()

    



############THE FOLLOWING CODE IS FOR THE ACTUAL EXECUTION AND IT IS JUST TEMPORARY#################
#Folder where the files for the current well are:
pathStr = "/path/to/your/folder"

def fireIt():
    InspectionWindow(pathStr)

#Create the main Tk window object with a Button that if pressed calls the function "fireIt"
#that creates an instance of "InspectionWindow" using the path written in "pathStr":
mainRoot = tk.Tk()
mainRoot.title("Folder Selection Window")
fireItButton = tk.Button(mainRoot, text = "Fire!!", command = fireIt)
fireItButton.pack()
mainRoot.mainloop()



