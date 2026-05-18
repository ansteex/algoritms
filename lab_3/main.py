import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from task1 import *
from task2 import *
from task3 import *
from task4 import *
def main():
    test_1d_approximation()
    test_1d_approximation1()
    #demonstrate_weights_effect()

    demonstrate_2d()

    demonstrate_nonlinear()

    demonstrate_ode()

if __name__ == "__main__":
    main()
