import matplotlib.pyplot as plt
import circlify
import numpy as np

# 替换下面的数据为您的节点和关系数据
labels = ['Node A', 'Node B', 'Node C']
relationships = np.array([[0, 0.5, 0.8],
                          [0.5, 0, 0.3],
                          [0.8, 0.3, 0]])

# 根据关系数据创建 Chord Diagram
circles = circlify.circlify([1, 1, 1])
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

for i, (label, circle) in enumerate(zip(labels, circles)):
    x, y, r = circle
    ax.annotate(label, (x, y), va='center', ha='center', fontsize=12)

for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j:
            strength = relationships[i, j]
            ax.plot([circles[i].x, circles[j].x], [circles[i].y, circles[j].y], linewidth=strength*10, color='grey', alpha=0.7)

plt.title('Chord Diagram - Node Relationships')
plt.show()
