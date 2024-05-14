#!/usr/bin/python

import os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def rpolarizabilities(fname):

    polas = []
    with open(fname) as pfname:

        for line in pfname:
            if line.find("Polarizability Tensor") >= 0:
                line = next(pfname)

                line = next(pfname)
                words = line.split()
                tmp = [float(words[0]), float(words[1]), float(words[2])]

                line = next(pfname)
                words = line.split()
                tmp += [float(words[0]), float(words[1]), float(words[2])]

                line = next(pfname)
                words = line.split()
                tmp += [float(words[0]), float(words[1]), float(words[2])]

                tmp = np.array(tmp).reshape(3, 3)
                tmp = (tmp + tmp.transpose()) / 2.

                polas.append([tmp[0][0], tmp[0][1], tmp[0][2], tmp[1][1], tmp[1][2], tmp[2][2]]) # 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz

    return np.array(polas)

def rxyz(fname):

    xyzs = []
    with open(fname) as pfname:
        for line in pfname:

            words = line.split()
            natms = int(words[0])

            line = next(pfname)

            xyz = []
            for i in range(natms):
                line = next(pfname)
                words = line.split()
                xyz.append([float(words[1]), float(words[2]), float(words[3])])

            xyz = np.array(xyz)
            xyzs.append(xyz.flatten())

    return xyzs

def main():

    train_size = 0.1 # for 10%
    # random state set to 1 for reproducible output

    sys.stderr.write("read XYZ coordinates from traj2\n")
    xyzs = rxyz("XYZ-traj2.xyz")
    npts = len(xyzs)

    sys.stderr.write("standardize XYZ coordinates\n")
    xyzs = StandardScaler().fit_transform(xyzs)

    sys.stderr.write("read polarizability tensors from traj2\n")
    polarizabilities = rpolarizabilities("POL-traj2.dat")

    sys.stderr.write("split training / test sets (%i/%i)\n"%(train_size*100., (1.-train_size)*100.))
    X_train, X_test, y_train, y_test = train_test_split(xyzs, polarizabilities, train_size=train_size, random_state=1)

    sys.stderr.write("train MLP\n")
    clf = MLPRegressor(hidden_layer_sizes=(100),random_state=1, max_iter=600).fit(X_train, y_train)

    sys.stdout.write("# SCORE R**2: TRAIN %12.6f (%12i SAMPLES) TEST %12.6f (%12i SAMPLES)\n"%(clf.score(X_train, y_train), train_size*npts, clf.score(X_test, y_test), (1.-train_size)*npts))

    sys.stderr.write("read XYZ coordinates from traj1\n")
    xyzs = rxyz("XYZ-traj1.xyz")

    sys.stderr.write("standardize XYZ coordinates\n")
    xyzs = StandardScaler().fit_transform(xyzs)

    for pol in clf.predict(xyzs):
        sys.stdout.write("Polarizability Tensor (Angs^3)\n\n")
        sys.stdout.write("%16.6f%16.6f%16.6f\n"%(pol[0], pol[1], pol[2]))
        sys.stdout.write("%16.6f%16.6f%16.6f\n"%(pol[1], pol[3], pol[4]))
        sys.stdout.write("%16.6f%16.6f%16.6f\n"%(pol[2], pol[4], pol[5]))

    return 0

if __name__ == "__main__":

    main()

