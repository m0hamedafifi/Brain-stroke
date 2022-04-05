import socket,os
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '127.0.0.1'
client_socket.connect((host, 5000))
k = ' '
size = 1024

try:
    class GUI:
        def run_form(self):
            window = Tk()
            window.title("BRAIN STROKE")
            window.geometry('500x500')

            def select_image():
                path = tkinter.filedialog.askopenfilename()
                if len(path) > 0:
                    # load the image from disk, convert it to grayscale, and detect
                    # edges in it
                    img = cv2.imread(path)
                    fig = plt.figure(figsize=(3, 3))
                    ax = fig.add_subplot(111)
                    ax.imshow(img, cmap='gray')
                    # plt.show()
                    show_img = FigureCanvasTkAgg(fig, window)
                    show_img.get_tk_widget().place(x=90, y=130)

                    myfile = open(path, 'rb')
                    print(myfile)
                    bytes = myfile.read()
                    print(bytes)
                    size = len(bytes)
                    size = "SIZE %s" % size
                    print(size)
                    # send image size to server
                    client_socket.sendall(size.encode("utf-8"))
                    answer = client_socket.recv(4096)

                    print('answer = %s' % answer)

                    # send image to server
                    print("bytes" + str(bytes))
                    client_socket.sendall(bytes)

                    # check what server send
                    answer = client_socket.recv(4096)
                    print('answer = %s' % answer)
                    diagnose = client_socket.recv(4096)
                    print('answer = %s' % diagnose)
                    tmp = str(diagnose).split("'")
                    diagnose = tmp[1].split("'")
                    print(diagnose[0])

                    precent = client_socket.recv(4096)
                    print('answer = %s' % precent)
                    tmp = str(precent).split("'")
                    precent = tmp[1].split("'")
                    print(precent[0])

                    print('Image successfully send to server')

                    label_feature2 = Label(window,
                                           text="This is a " + str(diagnose[0]) + "\n Brain " + str(precent[0]) + " %",
                                           font='Helvetica 18 bold')
                    label_feature2.place(x=165, y=440)

                    myfile.close()


            # initialize the window toolkit along with the two image
            label_feature = Label(window,text="AI Techniques for brain stroke \n telemedicine system",font='Helvetica 18 bold')
            label_feature.place(x=72, y=10)
            btn = Button(window, text="Select an image", command=select_image)
            btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
            btn.grid(column=2, row=0)
            btn.place(x=190, y=100)
            # kick off the GUI
            window.mainloop()
finally:
    g=GUI()
    g.run_form()
client_socket.close()

