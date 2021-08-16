import os


name1 = 0

while True:
    try:
        os.remove("../picture/" + str(name1) + ".jpg")
        name1 += 1
    except:
        break

f = open("../atoy/result.txt", 'w')
f.close()
f = open("../atoy/name.txt", 'w')
f.close()
f = open("../atoy/confidence.txt", 'w')
f.close()
print("finished")
