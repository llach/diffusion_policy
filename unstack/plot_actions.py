import os
import json
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


with open("ep1.pkl", "rb") as f:
    episode = pickle.load(f)

with open("actions.pkl", "rb") as f:
    actions = np.array(pickle.load(f))[:,1,:]

# Create a 3D plot
fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111, projection='3d')

# iterate over episodes
pos = episode["action"][2:]

# Create line segments for 3D
segments = np.array([pos[i:i+2] for i in range(len(pos)-1)])

# Normalize the values for color mapping
norm = Normalize(vmin=0, vmax=1)

# Plot each segment with the corresponding color
# for i, (start, end) in enumerate(segments):
#     color = plt.cm.viridis(norm(t_norm[i]))  # Use the colormap
#     ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=2)

ax.scatter(pos[:,0], pos[:,1], pos[:,2])
ax.scatter(actions[:,0], actions[:,1], actions[:,2])

# Scatter the points specified in stps in different colors with labels for the legend
scatter_colors = ['blue']#, 'green', 'orange']  # Colors for the scatter points
labels = ['close']#, 'end', 'open']  # Labels for the legend
# for idx, color, label in zip(scatter_colors, labels):
#     ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2], color=color, s=50, label=label+ f" [{idx}] ")


# Adjust the axes limits
ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

# Add a colorbar
# mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
# mappable.set_array(t_norm)

# plt.colorbar(mappable, ax=ax, shrink=0.5)

fig.tight_layout()
plt.show()