import numpy as np

def block_downsample(big):
    N = len(big)
    n = N/2
    out = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            i2 = 2*i
            j2 = 2*j
            out[i][j] = 0.25*(big[i2][j2]+big[i2+1][j2]+big[i2][j2+1]+big[i2+1][j2+1])

    return out

master = open('data.csv',"r")

raw_x = []
raw_y = []

for line in master:
    data = line.strip().split(",")
    print data
    contrast = data[2]
    if contrast == "True":
        raw_y.append(1)
    else:
        raw_y.append(0)

    filename = "../image_data/"+data[6]+".txt"

    xdata = np.loadtxt(filename)
    xdata = block_downsample(xdata)

    raw_x.append(xdata)

raw_x = np.array(raw_x)
raw_y = np.array(raw_y)

np.save("raw_x.npy",raw_x)

np.savetxt("raw_y.dat",raw_y)

