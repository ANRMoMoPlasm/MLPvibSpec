# MLPvibSpec
MLPvibSpec is a suite of Python tools to train a multi-layer perceptron model to simulate dipole moments and polarizability tensors from an AIMD trajectory.

# ML-dip.py
ML-dip.py takes as input cartesian coordinates (XYZ-traj1.xyz) and dipole moments (DIP-traj1.dat) from a trajectory, and outputs dipole moments corresponding to another trajectory (XYZ-traj2.xyz).

How to run: python ML-dip.py > DIP-traj2.ml

Multi-Layer Perceptron Model taking 1 hidden layer of 100 neurons.
![alt text](https://github.com/ANRMoMoPlasm/MLPvibSpec/blob/main/benchmark/Figure_1.png)

Multi-Layer Perceptron Model taking 2 hidden layer of 100 neurons.
![alt text](https://github.com/ANRMoMoPlasm/MLPvibSpec/blob/main/benchmark/Figure_2.png)

Multi-Layer Perceptron Model taking 3 hidden layer of 100 neurons.
![alt text](https://github.com/ANRMoMoPlasm/MLPvibSpec/blob/main/benchmark/Figure_3.png)

# ML-pol.py
ML-pol.py takes as input cartesian coordinates (XYZ-traj2.xyz) and polarizability tensors (POL-traj2.dat) from a trajectory, and outputs polarizability tensors corresponding to another trajectory (XYZ-traj1.xyz).

How to run: python ML-pol.py > POL-traj1.ml

Multi-Layer Perceptron Model taking 1 hidden layer of 100 neurons.
![alt text](https://github.com/ANRMoMoPlasm/MLPvibSpec/blob/main/benchmark/Figure_4.png)

Multi-Layer Perceptron Model taking 2 hidden layer of 100 neurons.
![alt text](https://github.com/ANRMoMoPlasm/MLPvibSpec/blob/main/benchmark/Figure_5.png)

Multi-Layer Perceptron Model taking 3 hidden layer of 100 neurons.
![alt text](https://github.com/ANRMoMoPlasm/MLPvibSpec/blob/main/benchmark/Figure_6.png)
