import os

txt_files = []
os.chdir("/mnt/train2017" )
files = os.listdir(os.getcwd())
for filename in files:
    filenamenoext = filename[:filename.index(".")]
    if filename.endswith(".jpg"):
        filenamenoext = filename[:filename.index(".")]
        if str(filenamenoext+".txt") not in files:
            txt_files.append("/mnt/train2017/" + str(filenamenoext+".txt"))

for txt in txt_files:
    f = open(txt, "w")
    f.close()

