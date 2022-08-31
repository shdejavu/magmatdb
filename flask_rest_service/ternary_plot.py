import matplotlib.pyplot as plt
import numpy as np
import ternary
import random

def system_name(X,Y,Z):
    formula = X + '-' + Y + '-' + Z
    return formula

def random_points(num_points, scale):
    points = []
    for i in range(num_points):
        x = random.randint(1, scale)
        y = random.randint(0, scale - x)
        z = scale - x - y
        points.append((x,y,z))
    return points
