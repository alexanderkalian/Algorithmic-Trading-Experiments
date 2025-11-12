'''
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
# PDF
doc = DocumentFile.from_images("test_screenshot_cropped2.png")
# Analyze
result = model(doc)

json_output = result.export()

print(json_output)
'''

import cv2
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import table_detection, table_recognition
from doctr.utils.visualization import draw_boxes

# 1) Read your image into a docTR "DocumentFile"
doc = DocumentFile.from_images("test_screenshot_cropped2.png")

# 2) Load the detection model
det_model = table_detection.table_detector(
    # This could be something like "model='sd_resnet50'"
    pretrained=True
)

# 3) Detect the table (or tables) in the doc
detection_result = det_model(doc)

# 4) Now load the structure recognition model
reco_model = table_recognition.table_recognizer(
    # Something like "model='master'", or relevant table-structure model
    pretrained=True
)

# 5) Perform recognition on the detected table(s)
recognition_result = reco_model(detection_result)

# 6) Access the output data (rows, columns, text)
#    recognition_result is docTR’s data structure with table info
for page_idx, page in enumerate(recognition_result.pages):
    print(f"== Page {page_idx+1} has {len(page.tables)} table(s) ==")
    for t_idx, table in enumerate(page.tables):
        print(f"--- Table {t_idx+1} ---")
        for cell in table.cells:
            # Each cell is typically (row_idx, col_idx, row_span, col_span, text)
            print(cell)

'''
from doctr.io import DocumentFile
from doctr.models import table_recognition

# 1) Load the image
doc = DocumentFile.from_images("test_screenshot_cropped2.png")

# 2) Get a pre-trained table recognition predictor
table_model = table_recognition.table_predictor(pretrained=True)

# 3) Run inference
table_result = table_model(doc)

# 4) Inspect the recognized tables
#    Each page in table_result.pages may have zero, one, or multiple tables
for page_idx, page in enumerate(table_result.pages):
    print(f"== Page {page_idx+1} contains {len(page.tables)} table(s) ==")
    for table_idx, table in enumerate(page.tables):
        print(f"--- Table {table_idx+1} ---")
        
        # The table's 'cells' are row/col-based
        # You can iterate over them to reconstruct the final CSV, JSON, etc.
        # docTR also often provides a 2D grid representation (page.tables[0].grid)
        
        if hasattr(table, "cells"):
            # (row, col, row_span, col_span, text)
            for cell in table.cells:
                print(cell)
        else:
            print("No table cells detected.")
'''