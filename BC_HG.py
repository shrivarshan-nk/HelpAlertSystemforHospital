import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
from tkinter import *
import dlib 
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.models import load_model

class BL():
    def eye_aspect_ratio(self,eye):
    	A = distance.euclidean(eye[1], eye[5])
    	B = distance.euclidean(eye[2], eye[4])

    	C = distance.euclidean(eye[0], eye[3])
    	eye = (A + B) / (2.0 * C)

    	return eye
    def __init__(self):
           global blinkCounter
           
           cap = cv2.VideoCapture(0)



           detector = dlib.get_frontal_face_detector()
           predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
           count = 0
           total = 0
           start=time.time()
           diff=0
           while True:
               ret, frame = cap.read()
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               
               now=time.time()
               curdif=int((now-start))
               if(curdif!=diff):
                   print(curdif)
                   diff=curdif
               if(curdif==5):
                   break
               
               faces = detector(gray)
               for face in faces:
                   landmarks = predictor(gray,face)

                   landmarks = face_utils.shape_to_np(landmarks)
                   leftEye = landmarks[42:48]
                   rightEye = landmarks[36:42]

                   leftEye = self.eye_aspect_ratio(leftEye)
                   rightEye = self.eye_aspect_ratio(rightEye)

                   eye = (leftEye + rightEye) / 2.0

                   if eye<0.3:
                       count+=1
                   else:
                       if count>=3:
                           total+=1

                       count=0
                   
               cv2.putText(frame, "Blink Count: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               cv2.imshow('Video',frame)
               if cv2.waitKey(1) & 0xff==ord('q'):
                   break
           cap.release()
           print(total)
           cv2.destroyAllWindows()
           blinklist={1:"Get Water",2:"Get Food",3:"Restroom Help",4:"Call Emergency"}
           if total<5:
               choice=blinklist.get(total)
           else:
               choice=blinklist.get(4)
           res(choice)  
           
class Hg:
    def __init__(self):
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils
    
        # Load the gesture recognizer model
        model = load_model('mp_hand_gesture')
    
        # Load class names
        f = open('gesture.names', 'r')
        classNames = f.read().split('\n')
        f.close()
        print(classNames)
    
    
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        start=time.time()
        diff=0
        instdif=0
        className = ''
        prevclassName=''
        ex=0
        while True:
            now=time.time()
        
            curdif=int((now-start))
            if(curdif==10):
                ex=1
                break
            # Read each frame from the webcam
            _, frame = cap.read()
    
            x, y, c = frame.shape
    
            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
            # Get hand landmark prediction
            result = hands.process(framergb)
    
            # print(result)
            
            
            changed=1
            
            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
    
                        landmarks.append([lmx, lmy])
    
                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    
                    # Predict gesture
                    prediction = model.predict([landmarks])
                    # print(prediction)
                    classID = np.argmax(prediction)
                    
                    prevclassName=className
                    
                    className = classNames[classID]
                    
                    if className!=prevclassName:
                        insttime=time.time()
                        changed=1
                    else:
                        changed=0
            
            if changed==0:
                 insdif=int((now-insttime))
                 print(insdif)
                 if(insdif==2):
                     
                    break        
                    
            # show the prediction on the frame
            
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0,0,255), 2, cv2.LINE_AA)
            
            
            # Show the final output
            cv2.imshow("Output", frame) 
            if cv2.waitKey(1) == ord('q'):
                break
    
        # release the webcam and destroy all active windows
        cap.release()
    
        cv2.destroyAllWindows() 
        hglist={'okay':"Get Water",
        'peace':"Restroom Help",
        'thumbs up':"Food",
        'thumbs down':"Dressing Aid",
        'call me':"Emergency",
        'stop':"Emergency",
        'one':"Standing Support",
        'fist':"Sitting Support",
        'smile':"Morale Support"}
        if ex!=1:
            choice=hglist.get(className) 
        else:
            choice="HELP"
        res(choice)   
           
hglist={'okay':"Get Water",
'peace':"Restroom Help",
'thumbs up':"Food",
'thumbs down':"Dressing Aid",
'call me':"Emergency",
'stop':"Emergency",
'one':"Standing Support",
'fist':"Sitting Support",
'smile':"Morale Support"}           
          
blinklist={1:"Get Water",2:"Get Food",3:"Restroom Help",4:"Call Emergency"}

def res(choice):
    global second
    second = Toplevel()
    second.title("Alert Window")
    second.geometry("500x300")
    second.configure(bg="#6495ED")
    lb1=Label(second,text="Alert Message!!!".format(choice),font=("Arial Bold", 20), bg="#6495ED",fg="red").pack(pady=10) 
    lb2=Label(second,text="You Called for:",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10)
    lb3=Label(second,text="{}".format(choice),font=("Arial Bold", 20), bg="#6495ED",fg="red").pack(pady=10)
    btn_back = Button(second, text="Back", font=("Arial", 14), bg="#FF8C00", fg="white",command=second.withdraw).pack(pady=10)

def instchoice():
    global ch
    ch=Toplevel()
    ch.title("Instruction Window")
    ch.geometry("500x300")
    ch.configure(bg="#6495ED")
    lb1=Label(ch,text="Instructions to View:",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10) 
    btn_enter = Button(ch, text="EyeBlinks", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:insbl(blinklist) )
    btn_enter.pack(pady=20)

    btn_enter1 = Button(ch, text="HandGesture", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:inshg(hglist) )
    btn_enter1.pack(pady=20)
    
    btn_back = Button(ch, text="Back", font=("Arial", 14), bg="#FF8C00", fg="white",command=ch.withdraw).pack(pady=10)

def insbl(blst):
    global inst
    inst=Toplevel()
    inst.title("Instruction Window")
    inst.geometry("500x400")
    inst.configure(bg="#6495ED")
    
    lb1=Label(inst,text="Instructions!!!",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10) 
    lb2=Label(inst,text="Blinks\t \tMeaning",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10)
    frame=Frame(inst,bg="#6495ED")
    frame.pack()
    frame2=Frame(frame, bg="#6495ED")
    frame2.pack(side=LEFT)
    frame3=Frame(frame, bg="#6495ED")
    frame3.pack(side=RIGHT,padx=38)
    for i in range(4):
        for j in range(2):
            if j==0:
                lb3=Label(frame2,text=list(blst.keys())[i],font=("Arial", 20), bg="#6495ED").pack(padx=105) 
            else:
                lb3=Label(frame3,text=list(blst.values())[i],font=("Arial", 20), bg="#6495ED").pack() 
    btn_edit= Button(inst,text="Edit",font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:[edit(blst) , inst.withdraw()]).pack(pady=10)  
    btn_back = Button(inst, text="Back", font=("Arial", 14), bg="#FF8C00", fg="white",command=inst.withdraw).pack(pady=10)    


def inshg(hglist):
    global inst
    inst=Toplevel()
    inst.title("Instruction Window")
    inst.geometry("500x600")
    inst.configure(bg="#6495ED")
    
    lb1=Label(inst,text="Hand Gesture Instructions!!!",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10) 
    lb2=Label(inst,text="Gesture\t\tMeaning",font=("Arial Bold", 20), bg="#6495ED").pack(pady=8)
    frame=Frame(inst,bg="#6495ED")
    frame.pack()
    frame2=Frame(frame, bg="#6495ED")
    frame2.pack(side=LEFT)
    frame3=Frame(frame, bg="#6495ED")
    frame3.pack(side=RIGHT,padx=38)
    for i in range(9):
        for j in range(2):
            if j==0:
                lb3=Label(frame2,text=(list(hglist.keys())[i]).capitalize(),font=("Arial", 20), bg="#6495ED").pack(padx=10) 
            else:
                lb3=Label(frame3,text=list(hglist.values())[i],font=("Arial", 20), bg="#6495ED").pack() 
    btn_edit= Button(inst,text="Edit",font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:[edithg(hglist) , inst.withdraw()]).pack(pady=10)  
    btn_back = Button(inst, text="Back", font=("Arial", 14), bg="#FF8C00", fg="white",command=inst.withdraw).pack(pady=10)
    
def edit(blst):
    global ed
    ed=Toplevel()
    ed.title("Edit Instruction Window")
    ed.geometry("500x500")
    ed.configure(bg="#6495ED")
    lb1=Label(ed,text="Edit Instructions!!!",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10) 
    lb2=Label(ed,text="Blinks\t \tMeaning",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10)
    frame=Frame(ed,bg="#6495ED")
    frame.pack()
    frame2=Frame(frame, bg="#6495ED")
    frame2.pack(side=LEFT)
    frame3=Frame(frame, bg="#6495ED")
    frame3.pack(side=RIGHT,padx=38)
    tf1=Entry(frame3,font=("Arial", 20))
    tf2=Entry(frame3,font=("Arial", 20))
    tf3=Entry(frame3,font=("Arial", 20))
    tf4=Entry(frame3,font=("Arial", 20))
    tf=[tf1,tf2,tf3,tf4]
    for i in range(4):
        for j in range(2):
            if j==0:
                lb3=Label(frame2,text=list(blst.keys())[i],font=("Arial", 20), bg="#6495ED").pack(padx=105) 
            else:
                txt=str(list(blst.values())[i])
                tf[i].pack()
                tf[i].insert(0,txt)        
    lb=Label(ed,text="",font=("Arial Bold", 20), bg="#6495ED")
    lb.pack(pady=10)
    btn_ok=Button(ed, text="OK", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:check(blst,tf,lb)).pack(pady=10)            
    btn_back = Button(ed, text="Back", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:[ed.withdraw(),insbl(blst)]).pack(pady=10)

def check(blst,tf,lb):
    global blinklist
    for i in range(4):
        txt=tf[i].get()
        blst[list(blst.keys())[i]]=txt
    for i in range(4):
        for j in range(4):
            if ((list(blst.values())[i].lower()==list(blst.values())[j].lower()) and (i!=j)):
               txt1="Invalid! matching meaning"
               break
            else:
               txt1="Successfully Modified"
        if txt1=="Invalid! matching meaning" :
            break
    lb.config(text=txt1)
    if txt1=="Successfully Modified":           
        for i in range(4):
            list(blinklist.values())[i]=list(blst.values())[i]
            
def edithg(blst):
    global ed
    ed=Toplevel()
    ed.title("Edit Instruction Window")
    ed.geometry("500x800")
    ed.configure(bg="#6495ED")
    lb1=Label(ed,text="Edit Instructions!!!",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10) 
    lb2=Label(ed,text="Gesture\t \tMeaning",font=("Arial Bold", 20), bg="#6495ED").pack(pady=10)
    frame=Frame(ed,bg="#6495ED")
    frame.pack()
    frame2=Frame(frame, bg="#6495ED")
    frame2.pack(side=LEFT)
    frame3=Frame(frame, bg="#6495ED")
    frame3.pack(side=RIGHT,padx=38)
    tf1=Entry(frame3,font=("Arial", 20))
    tf2=Entry(frame3,font=("Arial", 20))
    tf3=Entry(frame3,font=("Arial", 20))
    tf4=Entry(frame3,font=("Arial", 20))
    tf5=Entry(frame3,font=("Arial", 20))
    tf6=Entry(frame3,font=("Arial", 20))
    tf7=Entry(frame3,font=("Arial", 20))
    tf8=Entry(frame3,font=("Arial", 20))
    tf9=Entry(frame3,font=("Arial", 20))
    tf=[tf1,tf2,tf3,tf4,tf5,tf6,tf7,tf8,tf9]
    for i in range(9):
        for j in range(2):
            if j==0:
                lb3=Label(frame2,text=list(blst.keys())[i],font=("Arial", 20), bg="#6495ED").pack(padx=10) 
            else:
                txt=str(list(blst.values())[i])
                tf[i].pack()
                tf[i].insert(0,txt)        
    lb=Label(ed,text="",font=("Arial Bold", 20), bg="#6495ED")
    lb.pack(pady=10)
    btn_ok=Button(ed, text="OK", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:checkhg(blst,tf,lb)).pack(pady=10)            
    btn_back = Button(ed, text="Back", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:[ed.withdraw(),inshg(blst)]).pack(pady=10)

def checkhg(blst,tf,lb):
    global hglist
    for i in range(9):
        txt=tf[i].get()
        blst[list(blst.keys())[i]]=txt
    for i in range(9):
        for j in range(4):
            if ((list(blst.values())[i].lower()==list(blst.values())[j].lower()) and (i!=j)):
               txt1="Invalid! matching meaning"
               break
            else:
               txt1="Successfully Modified"
        if txt1=="Invalid! matching meaning" :
            break
    lb.config(text=txt1)
    if txt1=="Successfully Modified":           
        for i in range(9):
            list(hglist.values())[i]=list(blst.values())[i]

    
root = Tk()
root.title("Morse Code Detector")
root.geometry("500x400")


root.configure(bg="#6495ED") 

lbl_title = Label(root, text="Morse Code Detector", font=("Arial Bold", 20), bg="#6495ED")
lbl_title.pack(pady=10)


btn_enter = Button(root, text="Send EyeBlinks", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:BL() )
btn_enter.pack(pady=20)

btn_enter1 = Button(root, text="Send HandGesture", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:Hg() )
btn_enter1.pack(pady=20)


btn_edit = Button(root, text="Instructions", font=("Arial", 14), bg="#FF8C00", fg="white",command=lambda:instchoice())
btn_edit.pack(pady=10)


btn_exit = Button(root, text="Exit", font=("Arial", 14), bg="#FF8C00", fg="white", command=root.destroy)
btn_exit.pack(pady=10)
 

root.mainloop()