import os

image_files = []
os.chdir("/mnt500/GitHub/darknet/data/val2017" )
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("/mnt500/GitHub/darknet/data/val2017/" + filename)
os.chdir("..")
with open("/mnt500/GitHub/darknet/data/test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")