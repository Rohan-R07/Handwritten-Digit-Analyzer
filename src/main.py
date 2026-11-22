
import cv2
from src.preprocessing import preprocess_image, crop_digit


image_path = "test_images/testimg1.png"
padding = 10

original, grayscale, blurred, binary, cleaned = preprocess_image(
    image_path
)

try:
    cropped, info = crop_digit(cleaned, padding=padding)

    if info["touches_boundary"]:
        print(
            "Warning: digit touches the image boundary. Input may be "
            "clipped; missing portions are not reconstructed."
        )

    print("Digit bounding box:", info["bbox"])
    print("Padded crop box:", info["padded_bbox"])

    cv2.imshow("Original Image", original)
    cv2.imshow("Cleaned Binary Image", cleaned)
    cv2.imshow("Cropped Digit Image", cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except ValueError as exc:
    print(f"Error: {exc}")
