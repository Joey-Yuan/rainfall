import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

# Provided rainfall range data for Hong Kong districts in 2024
districts = [
    "Kwun Tong", "Eastern District", "Wong Tai Sin", "Kowloon City", "Southern District",
    "Central & Western District", "Sha Tin", "Sham Shui Po", "Wan Chai", "Yau Tsim Mong",
    "North District", "Tsuen Wan", "Sai Kung", "Tai Po", "Kwai Tsing", "Yuen Long",
    "Islands District", "Tuen Mun"
]
rainfall_min = [4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0]
rainfall_max = [5, 5, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3, 8, 6, 3, 3, 2, 1]

df = pd.DataFrame({
    'District': districts,
    'Rainfall Min (mm)': rainfall_min,
    'Rainfall Max (mm)': rainfall_max
})

# Animated pixel puddle visualization
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
text = ax.text(50, 95, "Click to see rainfall!", ha='center', fontsize=16)
info = None
im = None
ani = None  # store animation object globally

def create_puddle_array(size, layers):
    arr = np.zeros((100, 100), dtype=np.uint8)
    cx, cy = 50, 50
    for layer in range(layers):
        radius = int(size * (layer + 1) / layers)
        y, x = np.ogrid[-cx:100-cx, -cy:100-cy]
        mask = x*x + y*y <= radius*radius
        arr[mask] = 255 - layer * int(255/layers)
    return arr


def animate_puddle(rainfall, district, x=50, y=50):
    global im, info, ani
    if ani:
        if hasattr(ani, 'event_source') and ani.event_source is not None:
            ani.event_source.stop()  # stop previous animation
        ani = None
    if im:
        im.remove()
        im = None
    if info:
        info.remove()
        info = None
    size = 5 + rainfall * 2.5  # even smaller puddle size
    layers = 25  # even more ripple layers
    # Create exaggerated ripple arrays
    def create_ripple_array(size, layer, x, y, fade=1.0):
        arr = np.zeros((100, 100))
        rr, cc = np.ogrid[:100, :100]
        # Soft gradient center with noise
        dist = np.sqrt((rr-x)**2 + (cc-y)**2)
        center_radius = size * 0.5
        center_grad = np.clip(1 - dist/center_radius, 0, 1)
        # Add slight random noise for organic look
        noise = np.random.normal(0, 0.02, arr.shape)
        arr += (0.7 * fade * center_grad + noise) * (center_grad > 0)
        # Ripples, fading with layer
        for r in range(layer, layer+3):
            ripple_mask = ((rr-x)**2 + (cc-y)**2 <= (size + r*2)**2) & ((rr-x)**2 + (cc-y)**2 > (size + (r-1)*2)**2)
            arr[ripple_mask] = 0.5 * fade * (0.5 + 0.5 * np.sin(r/2))
        return arr
    def get_ripple_arrays(size, x, y, layers):
        # Precompute arrays for all frames with correct fading
        arrays = [create_ripple_array(size, l+1, int(x), int(y), fade=1-(l+1)/layers) for l in range(layers)]
        # Add a final frame that is completely transparent (all zeros)
        final = np.zeros((100, 100))
        arrays.append(final)
        return arrays
    ripple_arrays = get_ripple_arrays(size, x, y, layers)
    im = ax.imshow(ripple_arrays[0], cmap='Blues', alpha=0.7, extent=[0,100,0,100])
    # Create raindrop lines, more drops for higher rainfall
    num_drops = int(10 + rainfall * 10)
    # Wider spread for heavier rainfall
    spread_factor = 1.0 + min(rainfall/8, 1.5)
    drop_x = np.random.uniform(0, 100 * spread_factor, num_drops)
    drop_x = np.clip(drop_x, 0, 100)  # keep within bounds
    # Stagger starting y positions from top to bottom
    drop_y = np.linspace(100, 80, num_drops) + np.random.uniform(-5, 5, num_drops)
    drop_len = np.random.uniform(7, 15, num_drops)
    drop_lw = np.random.uniform(1.0, 2.5, num_drops)
    drop_color = np.random.choice(['deepskyblue', 'lightblue', 'dodgerblue', 'aqua'], num_drops)
    drop_lines = [ax.plot([x, x], [y, y+l], color=c, lw=w, alpha=0.6)[0] for x, y, l, w, c in zip(drop_x, drop_y, drop_len, drop_lw, drop_color)]

    def update(frame):
        fade = 1 - frame/layers
        artists = [im]
        if frame == len(ripple_arrays) - 1:
            im.set_visible(False)
            if info:
                info.set_alpha(0.0)
        else:
            puddle = create_ripple_array(size, frame+1, int(x), int(y), fade=0.7 * fade)
            im.set_data(puddle)
            im.set_alpha(0.7)
            if info:
                info.set_alpha(0.7 * fade)
            # Animate raindrops
            for i, line in enumerate(drop_lines):
                # Move drop down, add slight horizontal drift, reset if out of bounds
                drop_y[i] -= np.random.uniform(4, 7)
                drop_x[i] += np.random.uniform(-1.2, 1.2)
                if drop_y[i] < 0 or drop_x[i] < 0 or drop_x[i] > 100:
                    drop_y[i] = np.random.uniform(80, 100)
                    drop_x[i] = np.random.uniform(0, 100)
                    drop_len[i] = np.random.uniform(7, 15)
                    drop_lw[i] = np.random.uniform(1.0, 2.5)
                    drop_color[i] = np.random.choice(['deepskyblue', 'lightblue', 'dodgerblue', 'aqua'])
                    line.set_color(drop_color[i])
                    line.set_linewidth(drop_lw[i])
                line.set_xdata([drop_x[i], drop_x[i]])
                line.set_ydata([drop_y[i], drop_y[i]+drop_len[i]])
                line.set_alpha(0.6 * fade)
                artists.append(line)
        if info:
            artists.append(info)
        return artists
    ani = FuncAnimation(fig, update, frames=len(ripple_arrays), interval=80, blit=True, repeat=False)
    info = ax.text(x, y, f"{district}\nRainfall: {rainfall} mm", ha='center', va='center', fontsize=14, color='navy', weight='bold', alpha=0.7)
    fig.canvas.draw()

def on_click(event):
    if event.xdata is None or event.ydata is None:
        return
    idx = random.randint(0, len(districts)-1)
    rainfall = random.choice([rainfall_min[idx], rainfall_max[idx]])
    district = districts[idx]
    animate_puddle(rainfall, district, event.xdata, event.ydata)

fig.canvas.mpl_connect('button_press_event', on_click)
plt.title('Click to reveal rainfall in a random Hong Kong district (September 23, 2025 at 18:00)')
plt.show()


