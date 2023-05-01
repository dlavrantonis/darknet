import os

image_files = []
dirname = "/mnt/FalseDetectionsKalamatas/33/dataset/"
#dirname = "/mnt/FalseDetectionsPolemi/33/dataset/"

os.chdir(dirname )

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg") or filename.endswith(".png")or filename.endswith(".jpeg"):
        if not os.path.exists(dirname+filename[:filename.rfind(".")]+".txt"):
            with open(dirname+filename[:filename.rfind(".")]+".txt", "w") as outfile:
                #outfile.write("")
                outfile.close()

