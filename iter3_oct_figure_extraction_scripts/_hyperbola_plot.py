import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a, b = 3.0, 2.0
x = np.linspace(a, 10, 400)
y_pos = b * np.sqrt(x**2/a**2 - 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y_pos, 'b-', lw=2, label=r'$x^2/9 - y^2/4 = 1$')
ax.plot(x, -y_pos, 'b-', lw=2)
ax.plot(-x, y_pos, 'b-', lw=2)
ax.plot(-x, -y_pos, 'b-', lw=2)

t = np.linspace(-10, 10, 100)
ax.plot(t, (b/a)*t, 'r--', lw=1, label='asymptotes')
ax.plot(t, -(b/a)*t, 'r--', lw=1)

ax.plot([a, -a], [0, 0], 'ko', ms=6, label='vertices (±3, 0)')

ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)
ax.set_xlim(-10, 10)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Hyperbola: x²/9 − y²/4 = 1')
ax.legend()

plt.tight_layout()
plt.show()