import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import os
import fnmatch
import cv2
from matplotlib import image
from numpy import result_type
from signature import match, matchCropped, match_cli, match_cli1
from tkinter import *
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim

THRESHOLD = 85.0
ref_point = []


def browsefunc(ent):
    filename = askopenfilename(filetypes=([
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg"),
    ]))
    ent.delete(0, tk.END)
    ent.insert(tk.END, filename)


def verify_images(path1, path2):
    result = match(path1=path1, path2=path2)
    print(result)
    return result


def addstudent():
    root = tk.Tk()
    root.title("Signature Matching")
    root.configure(background='lightblue')
    root.geometry("700x700")  # 300x200
    uname_label = tk.Label(
        root, text="Signatures Verification & Recognition System:", font=10)
    uname_label.place(x=130, y=50)

    compare_button = tk.Button(
        root, text="Signature Recognition", font=10, command=recognition)
    compare_button.place(x=260, y=210)

    compare_button0 = tk.Button(
        root, text="Signature Extraction", font=10, command=extraction)
    compare_button0.place(x=260, y=310)

    compare_button1 = tk.Button(
        root, text="Signature Verification", font=10, command=verification)
    compare_button1.place(x=260, y=410)

    root.mainloop()
########################################


def extraction():
    def extract_sign(window, path1):
        image = cv2.imread(path1)
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([90, 38, 0])
        upper = np.array([145, 255, 255])
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        close = cv2.morphologyEx(
            opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        boxes = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x+w, y+h])

        boxes = np.asarray(boxes)
        left = np.min(boxes[:, 0])
        top = np.min(boxes[:, 1])
        right = np.max(boxes[:, 2])
        bottom = np.max(boxes[:, 3])

        result[close == 0] = (255, 255, 255)
        ROI = result[top:bottom, left:right].copy()
        cv2.rectangle(result, (left, top), (right, bottom), (36, 255, 12), 2)

        # cv2.imshow('result', result)
        # cv2.imshow('ROI', ROI)
        # cv2.imshow('close', close)
        cv2.imwrite("close.png", close)
        cv2.imwrite('result.png', result)
        cv2.imwrite('ROI.png', ROI)
        cv2.waitKey()

        images = {}
        users = os.listdir("./Users/")
        for i in users:
            images[i] = f"./Users/{i}/Primary/" + \
                os.listdir(f"./Users/{i}/Primary/")[0]
        print(images)
        best_match_val = -999
        best_match_key = None
        for k, v in images.items():
            score = match_cli1(ROI, v)
            print(score)
            if score > best_match_val:
                best_match_val = score
                best_match_key = k
        print(best_match_val, best_match_key)
        if best_match_val < THRESHOLD:
            messagebox.showerror(
                message="Failure: No Matching signature found", )
        else:
            messagebox.showinfo(
                message=f"Success: Signatures recognized, welcome {best_match_key}", )

    root = tk.Tk()
    root.title("Signature Extraction")
    root.geometry("700x700")  # 300x200
    root.configure(background='lightgreen')
    uname_label = tk.Label(
        root, text="Signatures Verification & Detection System:", font=10)
    uname_label.place(x=130, y=50)

    img1_message = tk.Label(root, text="Insert a document", font=10)
    img1_message.place(x=220, y=120)

    image1_path_entry = tk.Entry(root, font=10)
    image1_path_entry.place(x=210, y=160)

    img1_browse_button = tk.Button(
        root, text="Browse Signature", font=10, command=lambda: browsefunc(ent=image1_path_entry))
    img1_browse_button.place(x=240, y=200)

    verify_button = tk.Button(
        root, text="Signature Extraction & Detection", font=10, command=lambda: extract_sign(window=root,
                                                                                 path1=image1_path_entry.get(),))
    verify_button.place(x=245, y=300)

    root.mainloop()


def recognition():
    def recognise_user(window, path1):
        images = {}
        users = os.listdir("./Users/")
        for i in users:
            images[i] = f"./Users/{i}/Primary/" + \
                os.listdir(f"./Users/{i}/Primary/")[0]
        print(images)
        best_match_val = -999
        best_match_key = None
        for k, v in images.items():
            score = match_cli(path1, v)
            print(score)
            if score > best_match_val:
                best_match_val = score
                best_match_key = k
        print(best_match_val, best_match_key)
        if best_match_val < THRESHOLD:
            messagebox.showerror(
                message="Failure: No Matching signature found", )
        else:
            messagebox.showinfo(
                message=f"Success: Signatures recognized, welcome {best_match_key}", )

            listofimages = fnmatch.filter(
                os.listdir(f"./Users/{best_match_key}/"), "*.png")
            listofimages = [
                f"./Users/{best_match_key}/" + x for x in listofimages]
            print(listofimages)
            # canvas = tk.Canvas(root)
            # canvas.pack()
            # load = Image.open(listofimages[0]).resize(
            #     (960, 720), Image.ANTIALIAS)
            # w, h = load.size
            # render1 = ImageTk.PhotoImage(load)
            # image1 = canvas.create_image((w / 2, h / 2), image=render1)
            img1 = cv2.imread(listofimages[0])
            img2 = cv2.imread(listofimages[1])
            img3 = cv2.imread(listofimages[2])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
            # resize images for comparison
            img1 = cv2.resize(img1, (300, 300))
            img2 = cv2.resize(img2, (300, 300))
            img3 = cv2.resize(img3, (300, 300))

            cv2.imshow("Signature 1", img1)
            cv2.imshow("Signature 2", img2)
            cv2.imshow("Signature 3", img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #############################

    root = tk.Toplevel()
    root.title("User Recognition")
    root.geometry("700x700")  # 300x200
    root.configure(background='lightgreen')
    uname_label = tk.Label(
        root, text="Signatures Verification & Recognition System:", font=10)
    uname_label.place(x=130, y=50)

    img1_message = tk.Label(root, text="Put signature of a user", font=10)
    img1_message.place(x=220, y=120)

    image1_path_entry = tk.Entry(root, font=10)
    image1_path_entry.place(x=210, y=160)

    img1_browse_button = tk.Button(
        root, text="Browse Signature", font=10, command=lambda: browsefunc(ent=image1_path_entry))
    img1_browse_button.place(x=240, y=200)

    verify_button = tk.Button(
        root, text="User recognition", font=10, command=lambda: recognise_user(window=root,
                                                                               path1=image1_path_entry.get(),))
    verify_button.place(x=245, y=300)

    root.mainloop()

#############################

########################################


def verification():
    def verify_images_all(window, path1, choice):
        listofimages = fnmatch.filter(
            os.listdir(f"./Users/{choice}/"), "*.png")
        print(listofimages)
        bestmatch = []
        for i in listofimages:
            s = f"./Users/{choice}/{i}"
            m = match_cli(path1, s)
            bestmatch.append(m)
        bestmatch.sort(reverse=True)
        print(bestmatch)
        if bestmatch[0] < THRESHOLD:
            messagebox.showerror(
                message="Failure: Signatures is not matching", )
            pass
        else:
            messagebox.showinfo(
                message=f"Success: Signatures recognized, welcome {choice}", )

    def verification_user(choice):
        uname_label = tk.Label(
            root, text="User signature verification", font=10)
        uname_label.place(x=220, y=170)
        choice = variable.get()
        print(choice)

        img1_message = tk.Label(root, text="Put signature of user", font=10)
        img1_message.place(x=240, y=210)

        image1_path_entry = tk.Entry(root, font=10)
        image1_path_entry.place(x=220, y=250)

        img1_browse_button = tk.Button(
            root, text="Browse Signature", font=10, command=lambda: browsefunc(ent=image1_path_entry))
        img1_browse_button.place(x=245, y=300)

        verify_button = tk.Button(
            root, text="Signature Verification", font=10, command=lambda: verify_images_all(window=root, path1=image1_path_entry.get(),
                                                                                         choice=choice,))
        verify_button.place(x=237, y=360)

        ####################
    root = tk.Tk()
    root.title("Signature Matching")
    root.configure(background='#FFC300')
    root.geometry("700x700")  # 300x200
    uname_label = tk.Label(
        root, text="Signatures Detection and Verification System:", font=10)
    uname_label.place(x=130, y=50)

    variable = StringVar(root)
    flist = os.listdir("./Users/")
    print(flist)
    variable.set(flist[0])  # default value

    w = OptionMenu(root, variable, *flist, command=verification_user)
    w.pack(expand=True)
    w.place(x=300, y=120)

    root.mainloop()


#############################
root1 = Tk()
root1.title('Signature Detection and Verification System')
root1.geometry('1174x700+200+50')
root1.resizable(True, True)

IMAGE_PATH = 'Capture.jpg'
WIDTH, HEIGHT = 1300, 800

DataEntryFrame = Frame(root1, relief=GROOVE, borderwidth=5)
DataEntryFrame.place(x=10, y=30, width=1150, height=650)

img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize(
    (WIDTH, HEIGHT), Image.ANTIALIAS))
lbl = tk.Label(DataEntryFrame, image=img)
lbl.img = img
lbl.place(relx=0.5, rely=0.5, anchor='center')

# frontlabel = Label(DataEntryFrame, text='Signature Detection and Verification System', width=35,
#                   font=('arial', 23, 'italic bold'), bg='gold2')
#frontlabel.pack(side=TOP, expand=True)
addbtn = Button(DataEntryFrame, text='Signature Verification & Detection System', width=45, font=('arial', 20, 'bold'), bd=6, bg='skyblue3',
                activebackground='blue', relief=RIDGE,
                activeforeground='white', command=addstudent)
addbtn.pack(side=TOP, expand=True)


root1.mainloop()
