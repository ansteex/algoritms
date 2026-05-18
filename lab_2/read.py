import numpy as np
def read_3d_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    x_vals = [0, 1, 2, 3, 4]
    y_vals = [0, 1, 2, 3, 4]
    z_vals = []
    layers = []
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('z='):
            z = float(lines[i].split('=')[1])
            z_vals.append(z)
            i += 1
            if 'x' in lines[i].lower():
                i += 1
            layer = []
            for _ in range(5):
                row = [float(x) for x in lines[i].split()[1:6]]
                layer.append(row)
                i += 1
            layers.append(layer)
        else:
            i += 1
    
    order = np.argsort(z_vals)
    z_vals = [z_vals[i] for i in order]
    layers = [layers[i] for i in order]
    data = np.array(layers)
    
    return np.array(x_vals), np.array(y_vals), np.array(z_vals), data