import xycmap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

harvest = np.random.randn(4,4,1,2)

corner_colors = ("lightgrey", "green", "blue", "red")
n = (5, 5)  # x, y

cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)


fig, ax = plt.subplots()
im = ax.imshow(cmap)
plt.show()

