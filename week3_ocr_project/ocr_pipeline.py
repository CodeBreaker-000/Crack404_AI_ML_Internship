import os

# Temporary workaround for the OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import easyocr
import json


# --------------------------------------------------
# 1. PROJECT FOLDERS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_FOLDER = os.path.join(BASE_DIR, "raw_images")
GLOBAL_FOLDER = os.path.join(BASE_DIR, "cleaned_images", "global")
ADAPTIVE_FOLDER = os.path.join(BASE_DIR, "cleaned_images", "adaptive")
VIS_FOLDER = os.path.join(BASE_DIR, "visualized_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(GLOBAL_FOLDER, exist_ok=True)
os.makedirs(ADAPTIVE_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# 2. SETTINGS
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60


# --------------------------------------------------
# 3. EASYOCR READER
# --------------------------------------------------

reader = easyocr.Reader(["en"])


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, global_threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive_threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return image, gray, blurred, global_threshold, adaptive_threshold


# --------------------------------------------------
# 5. OCR TEXT EXTRACTION
# --------------------------------------------------

def extract_text(image_path):
    results = reader.readtext(image_path)

    detections = []

    for bbox, text, confidence in results:
        detection = {
            "text": str(text),
            "confidence": float(confidence),
            "bbox": [
                [float(point[0]), float(point[1])]
                for point in bbox
            ],
            "needs_review": bool(confidence < CONFIDENCE_THRESHOLD)
        }

        detections.append(detection)

    return detections


# --------------------------------------------------
# 6. CALCULATE AVERAGE CONFIDENCE
# --------------------------------------------------

def calculate_average_confidence(detections):
    if not detections:
        return 0.0

    confidences = [
        item["confidence"]
        for item in detections
    ]

    return sum(confidences) / len(confidences)


# --------------------------------------------------
# 7. DRAW BOUNDING BOXES
# --------------------------------------------------

def draw_bounding_boxes(
    image_path,
    detections,
    output_path,
    confidence_threshold=CONFIDENCE_THRESHOLD
):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    for item in detections:
        bbox = item["bbox"]
        confidence = item["confidence"]

        points = np.array(bbox, dtype=np.int32)

        if confidence >= confidence_threshold:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=color,
            thickness=2
        )

    cv2.imwrite(output_path, image)


# --------------------------------------------------
# 8. FIND RECEIPT IMAGES
# --------------------------------------------------

image_files = [
    file
    for file in os.listdir(RAW_FOLDER)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]

image_files.sort()

print(f"Found {len(image_files)} receipt images.")


# --------------------------------------------------
# 9. PROCESS ALL RECEIPTS
# --------------------------------------------------

all_results = []

for filename in image_files:
    print("\n" + "=" * 60)
    print(f"Processing: {filename}")
    print("=" * 60)

    image_path = os.path.join(RAW_FOLDER, filename)

    image, gray, blurred, global_img, adaptive_img = preprocess_image(image_path)

    global_path = os.path.join(GLOBAL_FOLDER, filename)
    adaptive_path = os.path.join(ADAPTIVE_FOLDER, filename)

    cv2.imwrite(global_path, global_img)
    cv2.imwrite(adaptive_path, adaptive_img)

    print("Preprocessing completed.")


    # ----------------------------------------------
    # OCR RAW IMAGE
    # ----------------------------------------------

    raw_detections = extract_text(image_path)
    raw_average = calculate_average_confidence(raw_detections)


    # ----------------------------------------------
    # OCR GLOBAL THRESHOLD IMAGE
    # ----------------------------------------------

    global_detections = extract_text(global_path)
    global_average = calculate_average_confidence(global_detections)


    # ----------------------------------------------
    # OCR ADAPTIVE THRESHOLD IMAGE
    # ----------------------------------------------

    adaptive_detections = extract_text(adaptive_path)
    adaptive_average = calculate_average_confidence(adaptive_detections)


    # ----------------------------------------------
    # SELECT BEST OCR METHOD
    # ----------------------------------------------

    methods = {
        "raw": {
            "detections": raw_detections,
            "average_confidence": raw_average,
            "image_path": image_path
        },
        "global": {
            "detections": global_detections,
            "average_confidence": global_average,
            "image_path": global_path
        },
        "adaptive": {
            "detections": adaptive_detections,
            "average_confidence": adaptive_average,
            "image_path": adaptive_path
        }
    }

    best_method = max(
        methods,
        key=lambda method: methods[method]["average_confidence"]
    )

    detections = methods[best_method]["detections"]
    average_confidence = methods[best_method]["average_confidence"]
    selected_image_path = methods[best_method]["image_path"]

    print(f"Selected OCR method: {best_method}")
    print(f"Selected average confidence: {average_confidence:.2%}")


    # ----------------------------------------------
    # PRINT DETECTED TEXT
    # ----------------------------------------------

    print("\nDetected text:")

    for item in detections:
        print(
            f"Text: {item['text']} | "
            f"Confidence: {item['confidence']:.2%} | "
            f"Needs Review: {item['needs_review']}"
        )


    # ----------------------------------------------
    # HIGH / LOW CONFIDENCE COUNTS
    # ----------------------------------------------

    high_confidence = sum(
        1
        for item in detections
        if item["confidence"] >= CONFIDENCE_THRESHOLD
    )

    low_confidence = sum(
        1
        for item in detections
        if item["confidence"] < CONFIDENCE_THRESHOLD
    )


    # ----------------------------------------------
    # PRINT CONFIDENCE COMPARISON
    # ----------------------------------------------

    print(f"\nRaw image average confidence: {raw_average:.2%}")
    print(f"Global threshold average confidence: {global_average:.2%}")
    print(f"Adaptive threshold average confidence: {adaptive_average:.2%}")
    print(f"High-confidence detections: {high_confidence}")
    print(f"Low-confidence detections: {low_confidence}")


    # ----------------------------------------------
    # SAVE BOUNDING BOX IMAGE
    # ----------------------------------------------

    visualized_path = os.path.join(VIS_FOLDER, filename)

    draw_bounding_boxes(
        selected_image_path,
        detections,
        visualized_path,
        CONFIDENCE_THRESHOLD
    )


    # ----------------------------------------------
    # STORE RESULT FOR THIS RECEIPT
    # ----------------------------------------------

    receipt_result = {
        "image": filename,
        "selected_method": best_method,
        "raw_average_confidence": float(raw_average),
        "global_average_confidence": float(global_average),
        "adaptive_average_confidence": float(adaptive_average),
        "selected_average_confidence": float(average_confidence),
        "high_confidence_count": int(high_confidence),
        "low_confidence_count": int(low_confidence),
        "detections": detections
    }

    all_results.append(receipt_result)


# --------------------------------------------------
# 10. OVERALL OCR METHOD SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL OCR METHOD SUMMARY")
print("=" * 60)

raw_count = 0
global_count = 0
adaptive_count = 0

for result in all_results:
    method = result["selected_method"]

    if method == "raw":
        raw_count += 1
    elif method == "global":
        global_count += 1
    elif method == "adaptive":
        adaptive_count += 1

print(f"Raw image selected: {raw_count} time(s)")
print(f"Global threshold selected: {global_count} time(s)")
print(f"Adaptive threshold selected: {adaptive_count} time(s)")


# --------------------------------------------------
# 11. TOTAL CONFIDENCE STATISTICS
# --------------------------------------------------

total_high_confidence = sum(
    result["high_confidence_count"]
    for result in all_results
)

total_low_confidence = sum(
    result["low_confidence_count"]
    for result in all_results
)

total_detections = total_high_confidence + total_low_confidence

print(f"\nTotal OCR detections: {total_detections}")
print(f"High-confidence detections: {total_high_confidence}")
print(f"Low-confidence detections: {total_low_confidence}")


# --------------------------------------------------
# 12. OVERALL AVERAGE CONFIDENCE
# --------------------------------------------------

if all_results:
    overall_average_confidence = sum(
        result["selected_average_confidence"]
        for result in all_results
    ) / len(all_results)
else:
    overall_average_confidence = 0.0

print(f"Overall average OCR confidence: {overall_average_confidence:.2%}")


# --------------------------------------------------
# 13. SAVE JSON RESULTS
# --------------------------------------------------

json_path = os.path.join(OUTPUT_FOLDER, "ocr_results.json")

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        all_results,
        file,
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# 14. SAVE TEXT REPORT
# --------------------------------------------------

text_report_path = os.path.join(OUTPUT_FOLDER, "ocr_report.txt")

with open(text_report_path, "w", encoding="utf-8") as report:
    report.write("OCR RECEIPT ANALYSIS REPORT\n")
    report.write("=" * 50 + "\n\n")

    for result in all_results:
        report.write(f"Receipt: {result['image']}\n")
        report.write(f"Selected method: {result['selected_method']}\n")
        report.write(
            f"Raw confidence: "
            f"{result['raw_average_confidence']:.2%}\n"
        )
        report.write(
            f"Global confidence: "
            f"{result['global_average_confidence']:.2%}\n"
        )
        report.write(
            f"Adaptive confidence: "
            f"{result['adaptive_average_confidence']:.2%}\n"
        )
        report.write(
            f"Selected confidence: "
            f"{result['selected_average_confidence']:.2%}\n"
        )
        report.write(
            f"High-confidence detections: "
            f"{result['high_confidence_count']}\n"
        )
        report.write(
            f"Low-confidence detections: "
            f"{result['low_confidence_count']}\n"
        )

        report.write("\nDetected Text:\n")

        for item in result["detections"]:
            if item["needs_review"]:
                status = "Human Review"
            else:
                status = "Accepted"

            report.write(
                f"- {item['text']} | "
                f"Confidence: {item['confidence']:.2%} | "
                f"{status}\n"
            )

        report.write("\n" + "-" * 50 + "\n\n")


    # ----------------------------------------------
    # OVERALL SUMMARY IN TEXT REPORT
    # ----------------------------------------------

    report.write("OVERALL SUMMARY\n")
    report.write("=" * 50 + "\n")

    report.write(f"Receipts processed: {len(all_results)}\n")
    report.write(f"Raw selected: {raw_count}\n")
    report.write(f"Global selected: {global_count}\n")
    report.write(f"Adaptive selected: {adaptive_count}\n")
    report.write(f"Total detections: {total_detections}\n")
    report.write(f"High-confidence detections: {total_high_confidence}\n")
    report.write(f"Low-confidence detections: {total_low_confidence}\n")

    report.write(
        f"Overall average confidence: "
        f"{overall_average_confidence:.2%}\n"
    )


# --------------------------------------------------
# 15. FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("OCR PIPELINE COMPLETED")
print("=" * 60)

print(f"Processed receipts: {len(image_files)}")
print(f"JSON results saved to: {json_path}")
print(f"Text report saved to: {text_report_path}")