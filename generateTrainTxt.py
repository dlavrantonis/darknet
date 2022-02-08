import os

image_files = []
dirname = "/mnt/FalseDetectionsPolemi/11/dataset/"
#dirname = "/mnt/FalseDetectionsKalamatas/7/dataset/"

os.chdir(dirname )

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_files.append(dirname + filename)

os.chdir("..")
#with open("/mnt500/GitHub/darknet/data/FalseDetectionsKalamatas7.txt", "w") as outfile:
with open("/mnt500/GitHub/darknet/data/FalseDetectionsPolemi11.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")