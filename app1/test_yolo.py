from yolo_predict import detect_image_yolo

# Test image path (update this path to a valid image file on your system)
img_path = "D:\\Registration system\\registration\\app1\\media_uploads\\test_image.jpg"

print("Testing YOLO Detection...")

try:
    results = detect_image_yolo(img_path)
    print("Detection Results:", results)
except Exception as e:
    print("Error during detection:", str(e))
