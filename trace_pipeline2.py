import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from preprocessing import preprocess_image, crop_digit, normalize_digit_image
from feature_extraction import extract_normalized_features
from digit_recognizer import recognize_digit

images_to_test = [
    ('test_images/digit2.png', 2),
    ('test_images/digit2another.png', 2),
    ('test_images/image3.png', 3),
    ('test_images/imagenormal3.png', 3),
]

for img_path, expected_digit in images_to_test:
    print('=' * 80)
    print(f'IMAGE: {img_path} (expected digit: {expected_digit})')
    print('=' * 80)
    
    try:
        image, gray, blurred, binary, cleaned = preprocess_image(img_path)
        cropped, info = crop_digit(cleaned, padding=10)
        norm = normalize_digit_image(cropped)
        
        # Compute what min_hole_area recognize_digit will use
        bbox_area = info['bbox'][2] * info['bbox'][3]
        computed_min_hole_area = max(5, int(0.001 * bbox_area))
        print(f'Bbox: {info["bbox"]}, area: {bbox_area}, min_hole_area: {computed_min_hole_area}')
        
        # Feature extraction with computed min_hole_area (what recognize_digit uses)
        nf = extract_normalized_features(cropped, crop_info=info, min_hole_area=computed_min_hole_area)
        print(f'Hole count (computed min_area): {nf["hole_count"]}')
        print(f'Hole positions: {nf["normalized_hole_positions"][:5]}...')
        print(f'Hole areas: {[round(a, 4) for a in nf["normalized_hole_areas"][:5]]}')
        print(f'Aspect ratio: {nf["aspect_ratio"]:.4f}')
        print(f'Foreground density: {nf["foreground_density"]:.4f}')
        
        # Full recognition
        result = recognize_digit(norm, crop_info=info, min_hole_area=20)
        print(f'FINAL: pred={result["predicted_digit"]}, status={result["status"]}, score={result["similarity_score"]:.4f}')
        print(f'Template scores: {[(d, round(s, 4)) for d, s in sorted(result["template_scores"].items(), key=lambda x: -x[1])[:5]]}')
        if result.get('feature_scores'):
            print(f'Feature scores: {[(d, round(s, 4)) for d, s in sorted(result["feature_scores"].items(), key=lambda x: -x[1])[:5]]}')
        print(f'Expected: {expected_digit}, Correct: {result["predicted_digit"] == expected_digit}')
        print(f'Explanation: {result["explanation"][:200]}')
        
    except Exception as e:
        import traceback
        print(f'ERROR: {type(e).__name__}: {e}')
        traceback.print_exc()
    
    print()
