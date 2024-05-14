#!/usr/bin/python

import os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def rdipole(fname):

    dipoles = []
    with open(fname) as pfname:

        for line in pfname:

            if line.find("DIPOLE MOMENT:") >= 0:
                words = line.split()
                dipole = np.array([float(words[3][1:-1]), float(words[4][0:-1]), float(words[5][0:-1])])

                dipoles.append(dipole)

            else:
                words = line.split()
                dipoles.append([float(words[0]), float(words[1]), float(words[2])])

    return np.array(dipoles)

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

    sys.stderr.write("read XYZ coordinates from traj1\n")
    xyzs = rxyz("XYZ-traj1.xyz")
    npts = len(xyzs)

    sys.stderr.write("standardize XYZ coordinates\n")
    xyzs = StandardScaler().fit_transform(xyzs)

    sys.stderr.write("read dipole moments from traj1\n")
    dipoles = rdipole("DIP-traj1.dat")

    sys.stderr.write("split training / test sets (%i/%i)\n"%(train_size*100., (1.-train_size)*100.))
    X_train, X_test, y_train, y_test = train_test_split(xyzs, dipoles, train_size=train_size, random_state=1)

    sys.stderr.write("train MLP\n")
    clf = MLPRegressor(hidden_layer_sizes=(100),random_state=1, max_iter=600).fit(X_train, y_train)

    sys.stdout.write("# SCORE R**2: TRAIN %12.6f (%12i SAMPLES) TEST %12.6f (%12i SAMPLES)\n"%(clf.score(X_train, y_train), train_size*npts, clf.score(X_test, y_test), (1.-train_size)*npts))

    sys.stderr.write("read XYZ coordinates from traj2\n")
    xyzs = rxyz("XYZ-traj2.xyz")

    sys.stderr.write("standardize XYZ coordinates\n")
    xyzs = StandardScaler().fit_transform(xyzs)

    for dip in clf.predict(xyzs):
        sys.stdout.write("QM  DIPOLE MOMENT: {%.6f, %.6f, %.6f} (|D| = %.6f) DEBYE\n"%(dip[0], dip[1], dip[2], np.sqrt(np.dot(dip, dip))))

    return 0

if __name__ == "__main__":

    main()

