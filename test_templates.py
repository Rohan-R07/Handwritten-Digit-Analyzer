import sys, cv2, numpy as np
sys.path.insert(0, 'src')
from digit_recognizer import recognize_digit

print('Template self-matching test:')
all_correct = True
for d in range(10):
    t = cv2.imread('templates/{}.png'.format(d), cv2.IMREAD_GRAYSCALE)
    r = recognize_digit(t, min_hole_area=20)
    correct = r['predicted_digit'] == d
    all_correct = all_correct and correct
    print('  Template {}: pred={}, status={}, score={:.4f}, correct={}'.format(
        d, r['predicted_digit'], r['status'], r['similarity_score'] or 0, correct))
print('All templates correct: {}'.format(all_correct))