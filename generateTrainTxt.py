import os

image_files = []
dirname = "/mnt/FalseDetectionsKalamatas/10/dataset/"
#dirname = "/mnt/FalseDetectionsPolemi/14/dataset/"

os.chdir(dirname )

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_files.append(dirname + filename)

os.chdir("..")
with open("/mnt500/GitHub/darknet/data/FalseDetectionsKalamatas10.txt", "w") as outfile:
#with open("/mnt500/GitHub/darknet/data/FalseDetectionsPolemi14.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")