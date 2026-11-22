import cv2
import numpy as np
import os

test_dir = 'test_images'
for f in sorted(os.listdir(test_dir)):
    path = os.path.join(test_dir, f)
    img = cv2.imread(path)
    if img is None:
        print(f'{f}: FAILED TO LOAD')
        continue
    h, w, c = img.shape
    unique_colors = len(np.unique(img.reshape(-1, c), axis=0))
    print(f'{f}: shape={img.shape}, channels={c}, unique_colors={unique_colors}')
    
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    print(f'  B range: [{b.min()}, {b.max()}], G range: [{g.min()}, {g.max()}], R range: [{r.min()}, {r.max()}]')
    is_gray = np.allclose(b, g) and np.allclose(g, r)
    print(f'  Is grayscale: {is_gray}')
    if not is_gray:
        b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
        print(f'  Mean BGR: [{b_mean:.1f}, {g_mean:.1f}, {r_mean:.1f}]')
        print(f'  Bluish: {b_mean > r_mean + 10}')
