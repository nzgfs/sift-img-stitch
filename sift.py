import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import WindowsPath
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

window=Tk()
window.title("sift")
window.geometry("900x750")

def getimg1():
    global file_path1
    file_path1 = filedialog.askopenfilename(title='select', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
    img = Image.open(file_path1)
    width, height = img.size
    if img.width/img.height>=4/3:   
        img = img.resize((400, int(400/width*height)))   
    else:
        img = img.resize((int(300/height*width), 300))   

    global photo1
    photo1 = ImageTk.PhotoImage(img)
    img01.configure(image = photo1,width=400,height=300)  
    img01.image = photo1

def getimg2():
    global file_path2
    file_path2 = filedialog.askopenfilename(title='select', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
    img = Image.open(file_path2)
    width, height = img.size
        
    if img.width/img.height>=4/3:
        img = img.resize((400, int(400/width*height)))
    else:
        img = img.resize((int(300/height*width), 300))

    global photo2
    photo2 = ImageTk.PhotoImage(img)
    img02.configure(image = photo2,width=400,height=300)
    img02.image = photo2

def stitchimg():
    imgA = cv2.imread(file_path1)
    imgB = cv2.imread(file_path2)
    result = stitch([imgA, imgB], showMatches=True)

    cv2.imwrite("result.jpg",result)
    re=Image.open("result.jpg")

    width, height = re.size
    if re.width/re.height>=43/15:
        re =  re.resize((860, int(860/width*height)))
    else:
        re =  re.resize((int(300/height*width), 300))
    global photo3
    photo3 = ImageTk.PhotoImage(re)
    img03.configure(image = photo3,width=860,height=300)
    img03.image = photo3

def save():
    filename = filedialog.asksaveasfile(defaultextension=".jpg")
    if not filename:
        return
    out=Image.open("result.jpg")
    out.save(filename)

def detect(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()   
    (kps, feat) = sift.detectAndCompute(grey, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, feat  

def match(kpsA, kpsB, featA, featB, ratio, reprojThresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")       
    knnMatches = matcher.knnMatch(featA, featB, 2)   #KNN
    matches = []

    for m in knnMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))   

    if len(matches) > 4:
        A = np.float32([kpsA[i] for (_, i) in matches])  
        B = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(A, B, cv2.RANSAC, reprojThresh)   
        return (matches, H, status)
    return None
    
def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    (imgB, imgA) = images
    (kpsA, featA) = detect(imgA)   
    (kpsB, featB) = detect(imgB)
    M = match(kpsA, kpsB, featA, featB, ratio, reprojThresh)  
    if M is None:   
        return None
    (matches, H, status) = M
    result = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))   
    result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB   
    return result

global photo1
global file_path1
photo1 = None
fr01 = Frame(window,width=400,height=300,background="white")
fr01.grid(column=0, row=0)
fr01.place(x=20,y=20)
img01=Label(fr01,image=photo1,width=56,height=17,background="white")
img01.pack()
btn01 = Button(window, text='Open',width=6,command=getimg1,relief=GROOVE)
btn01.grid(column=0, row=1)
btn01.place(x=200,y=332)

global photo2
global file_path2
photo2 = None
fr02 = Frame(window,width=400,height=300,background="white")
fr02.grid(column=1, row=0)
fr02.place(x=480,y=20)
img02=Label(fr02,image=photo2,width=56,height=17,background="white")
img02.pack()
btn02 = Button(window, text='Open',width=6,command=getimg2,relief=GROOVE)
btn02.grid(column=1, row=1)
btn02.place(x=650,y=332)

btn03=Button(window,text='Stitch',width=7,command=stitchimg,relief=GROOVE)
btn03.grid(column=2, row=1)
btn03.place(x=420,y=335)

global photo3
photo3 = None
fr03 = Frame(window,width=860,height=300,background="white")
fr03.grid(column=0, row=2)
fr03.place(x=20,y=376)
img03=Label(fr03,image=photo3,width=122,height=18,background="white")
img03.pack()

btn04=Button(window,text='Save',width=7,command=save,relief=GROOVE)
btn04.grid(column=2, row=3)
btn04.place(x=420,y=705)

window.mainloop()
