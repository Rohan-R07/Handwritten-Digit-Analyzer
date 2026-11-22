import sys, cv2, numpy as np
sys.path.insert(0, 'src')
from preprocessing import preprocess_image, crop_digit, normalize_digit_image
from feature_extraction import extract_normalized_features, extract_hole_features

image_path = 'test_images/0image.png'
image, gray, blurred, binary, cleaned = preprocess_image(image_path)
cropped, info = crop_digit(cleaned, padding=10)

# Directly call extract_hole_features
hf = extract_hole_features(cropped, min_hole_area=20)
print('extract_hole_features result:')
print('  num_enclosed_holes:', hf['num_enclosed_holes'])
print('  hole_areas:', hf['hole_areas'])
print('  hole_centers:', hf['hole_centers'])
print('  normalized_hole_positions:', hf['normalized_hole_positions'])

# Now call extract_normalized_features
nf = extract_normalized_features(cropped, crop_info=info, min_hole_area=20)
print()
print('extract_normalized_features result keys:', list(nf.keys()))
print('  hole_count:', nf.get('hole_count', 'MISSING'))
print('  normalized_hole_positions:', nf.get('normalized_hole_positions', 'MISSING'))
print('  normalized_hole_areas:', nf.get('normalized_hole_areas', 'MISSING'))
