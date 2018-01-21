import numpy as np
import matplotlib.pyplot as plt
import csv

data = [] #create empty list
write_file = []
with open('data/driving_log.csv', 'rU') as f:
    #reader = csv.reader(f, ' ', quoting=csv.QUOTE_NONNUMERIC)
    reader = csv.reader(f)
    first_line=True
    count=0
    for line in reader:
        if(first_line):
            first_line=False
        else:
            if float(line[3])==0 and count <4000:
                count+=1
            else:
                data.append(float(line[3]))
                write_file.append(line)

with open("data/driving_log_truncated.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(write_file)

print("Ignored these zeros:",count)
# generate the histogram
hist, bin_edges=np.histogram(data, bins=50, range=[-1, 1])


# generate histogram figure
plt.hist(data, bin_edges)
#plt.savefig('chart_file', format="pdf")
#plt.show()

from keras.utils import plot_model

from keras.models import load_model
import h5py

#model = load_model("model.h5")
#plot_model(model, to_file='model.png')


import cv2
name = 'straight.jpg'
center_image = cv2.imread(name)
center_image = cv2.flip(center_image,1)
cv2.imwrite('flip.jpg', center_image)

