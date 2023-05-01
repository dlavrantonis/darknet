import os

image_files = []
#dirname = "/mnt500/GitHub/darknet/data/crowdhuman-608x608/"
#dirname = "/mnt/FalseDetectionsKalamatas/33/dataset/"
dirname = "/mnt/FalseDetectionsPolemi/33/dataset/"
#dirname = "/mnt/theGreatDataset/labeled/"

os.chdir(dirname )

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
        image_files.append(dirname + filename)

os.chdir("..")
#with open("/mnt500/GitHub/yolov7/darknet/data/crowdhuman.txt", "w") as outfile:
#with open("/mnt500/GitHub/yolov7/darknet/data/FalseDetectionsKalamatas33.txt", "w") as outfile:
with open("/mnt500/GitHub/yolov7/darknet/data/FalseDetectionsPolemi33.txt", "w") as outfile:
#with open("/mnt500/GitHub/newdarknet/data/theGDLabeled.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")