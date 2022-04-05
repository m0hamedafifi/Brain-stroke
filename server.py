import socket
import os
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime

class GUI:
    sendandrec=False
    accpet_var=0
    connect_status = False
    clientsoc=NONE
    st = "Waiting For Connection"
    def run_form(self):
        window = Tk()
        window.title("BRAIN STROKE")
        window.geometry('500x500')

        label_feature = Label(window, text=self.st,font=('helvetica', 10, 'bold'))
        label_feature.place(x=0, y=3)
        print("iam out")
        timer=1000
        def clock():

            if (self.accpet_var == 1):
                #print("iam accpeted")
                self.clientsoc,self.st=connecting()
                self.connect_status=True
                self.accpet_var += 1
            if (self.connect_status == True):
                #print("iam in")
                label_feature = Label(window, text="from: "+self.st+"  ",font=('helvetica', 10, 'bold'))
                label_feature.place(x=0, y=3)
                self.connect_status = False
                self.sendandrec=True
            if(self.sendandrec==True):
                buffer_size = 4096
                basename = "1.jpg"

                img= datarec(self.clientsoc,buffer_size,basename,window)
                fig = plt.figure(figsize=(3, 3))
                ax = fig.add_subplot(111)
                ax.imshow(img, cmap='gray')
                # plt.show()
                show_img = FigureCanvasTkAgg(fig, window)
                show_img.get_tk_widget().place(x=90, y=160)
            # lab['text'] = time

            window.after(10000, clock)  # run itself again after 1000 ms

        clock()
        '''
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        # plt.show()
        show_img = FigureCanvasTkAgg(fig, window)
        show_img.get_tk_widget().place(x=90, y=160)
        '''
        #print(self.accpet_var)
        self.accpet_var+=1
        #print(self.accpet_var)

        window.mainloop()


def connecting():
    client_socket, address = server_socket.accept()
    print("Conencted to - ", address, "\n")
    return client_socket,str(address)
#buffer_size = 4096
#basename = "1.jpg"
def datarec(client_socket,buffer_size,basename,w):
    try:
        print(' Buffer size is %s' % buffer_size)
        data = client_socket.recv(buffer_size)
        print(data)

        txt = str(data)
        print(txt)
        if txt.startswith('b' + "'SIZE"):
            tmp = txt.split()
            tmp = tmp[1].split("'")
            print(tmp[0])
            size = int(tmp[0])

            print('got size')
            print('size is %s' % size)

            client_socket.sendall("GOT SIZE".encode("utf-8"))
            # Now set the buffer size for the image
            buffer_size = size + 2

        data = client_socket.recv(buffer_size)

        if data:
            myfile = open(basename, 'wb')

            # data = sock.recv(buffer_size)
            myfile.write(data)
            myfile.close()
            client_socket.sendall("GOT IMAGE".encode("utf-8"))
            buffer_size = 4096
            img = cv2.imread(basename)
            client_socket.sendall("Stroke".encode("utf-8"))
            client_socket.sendall("85".encode("utf-8"))

        return  img
    except:
        print("Client left")
        w.destroy()
        server_socket.close()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(("", 5000))
server_socket.listen(5)
print("welcome the server")

Win=GUI()
Win.run_form()
server_socket.close()




#os.system('python clint.py')

while (True):

                print(' Buffer size is %s' % buffer_size)
                data = client_socket.recv(buffer_size)
                print(data)

                txt = str(data)
                print(txt)
                if txt.startswith('b'+"'SIZE"):
                    tmp = txt.split()
                    print(tmp[0])
                    print(tmp[1])
                    tmp=tmp[1].split("'")
                    print(tmp[0])
                    size = int(tmp[0])

                    print ('got size')
                    print ('size is %s' % size)

                    client_socket.sendall("GOT SIZE".encode("utf-8"))
                    # Now set the buffer size for the image
                    buffer_size = size+2


                elif data:
                    myfile = open(basename, 'wb')

                    # data = sock.recv(buffer_size)
                    if not data:
                        myfile.close()
                        break
                    myfile.write(data)
                    myfile.close()
                    client_socket.sendall("GOT IMAGE".encode("utf-8"))
                    buffer_size = 4096
                    img = cv2.imread(basename)
                    g.run_form(ip=address,img=img)
                    client_socket.shutdown()

server_socket.close()
#os.system('python clint.py')
##########################
#################
############
''''
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
image = ImageTk.PhotoImage(image)
# if the panels are None, initialize them
if panelA is None:
    # the first panel will store our original image
    panelA = Label(image=image)
    panelA.image = image
    panelA.pack(side="left", padx=10, pady=10)
else:
    # update the pannels
    panelA.configure(image=image)
    panelA.image = image
'''