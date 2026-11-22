import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')
from preprocessing import preprocess_image, crop_digit, normalize_digit_image
from digit_recognizer import recognize_digit
from feature_extraction import extract_normalized_features
from template_matcher import match_template

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
        
        # Get raw template matching results
        tmpl = match_template(norm)
        print(f'Pixel scores (top 5): {[(d, round(s, 4)) for d, s in sorted(tmpl["pixel_scores"].items(), key=lambda x: -x[1])[:5]]}')
        print(f'Shape scores (top 5): {[(d, round(s, 4)) for d, s in sorted(tmpl["shape_scores"].items(), key=lambda x: -x[1])[:5]]}')
        print(f'Combined template scores (top 5): {[(d, round(s, 4)) for d, s in tmpl["ranked_candidates"][:5]]}')
        
        # Full recognition
        result = recognize_digit(norm, crop_info=info, min_hole_area=20)
        print(f'FINAL: pred={result["predicted_digit"]}, status={result["status"]}, score={result["similarity_score"]:.4f}')
        print(f'Expected: {expected_digit}, Correct: {result["predicted_digit"] == expected_digit}')
        print(f'Explanation: {result["explanation"][:200]}')
        
    except Exception as e:
        import traceback
        print(f'ERROR: {type(e).__name__}: {e}')
        traceback.print_exc()
    
    print()
