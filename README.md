Iranian Automatic License Plate Recognition (ALPR)
An end-to-end, PC-based Iranian license plate recognition pipeline built with YOLO and PyTorch. The system accepts vehicle images or image directories, localizes Iranian plates, detects character regions, classifies each character with a 34-class ResNet18, reconstructs the plate in the correct image-space order, and exports visual and structured results.

Demo pipeline
text

Image
  -> optional COCO vehicle detection
  -> Iranian plate detection (YOLO11)
  -> character / Iran-region detection (YOLO11)
  -> 34-class character recognition (ResNet18)
  -> geometric ordering and Iranian plate parsing
  -> annotated image + JSON + CSV
Features
Single-image and directory inference
Optional pretrained COCO vehicle detector
Dedicated Iranian plate detector
Dedicated char / iran region detector
Existing 34-class ResNet18 character classifier
Correct geometric ordering: six serial characters followed by two province digits
Persian and romanized plate output
Soft plate-grammar and province-code diagnostics
Partial readings retained when eight character boxes are unavailable
Annotated output images
Machine-readable JSON and CSV reports
CUDA acceleration when available, with CPU fallback
Models
The repository expects these local weights by default:

text

checkpoints/plate_yolo11s_v1.pt
checkpoints/resnet18_char34_v1.pth
runs/detect/runs/char/finetune_v3/weights/best.pt
Character-detector fallbacks:

text

checkpoints/char_yolo11s_ft_v2.pt
checkpoints/char_yolo11s_ft_v1.pt
The optional vehicle detector defaults to yolo11s.pt and uses the COCO car, motorcycle, bus, and truck classes.

Model checkpoints and datasets should not be committed to Git. Publish download instructions or release links separately if redistribution is permitted.

Installation
Python 3.10 or newer is recommended.

Bash

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-alpr.txt
For GPU inference, install a PyTorch build compatible with the local CUDA version before installing the remaining requirements.

Usage
Run commands from the repository root.

Vehicle or car-crop image
For a car crop, bypass COCO vehicle detection:

Bash

python run_alpr.py path/to/car.jpg --no-vehicle --output outputs/demo
Directory of car crops
Bash

python run_alpr.py path/to/images --no-vehicle --output outputs/demo
Full scene image
Keep vehicle detection enabled for a full traffic scene:

Bash

python run_alpr.py path/to/scene.jpg --output outputs/demo
Useful options
Bash

python run_alpr.py --help
Common thresholds:

text

--vehicle-conf 0.30
--plate-conf   0.25
--char-conf    0.20
--iran-conf    0.08
The detector input sizes default to 640 for vehicles, 960 for plates, and 640 for characters.

Output
For --output outputs/demo, the program creates:

text

outputs/demo/
├── images/          # annotated images
├── results.json     # full nested detection records
└── results.csv      # tabular summary
Each record includes source path, bounding boxes, class IDs, per-character confidence, Persian and romanized readings, completeness, grammar diagnostics, province diagnostics, and stage-level confidence values.

Evaluation snapshot
A functional integration test was run on 152 held-out car-crop images:

Metric	Result
Images with at least one detected plate	151 / 152 (99.34%)
Total plate detections	185
Complete eight-character detections	147 / 185 (79.46%)
Grammar-valid readings	110 / 185 (59.46%)
Known-province diagnostic	116 / 185 (62.70%)
Mean plate confidence	0.776
Mean character-detection confidence	0.732
Mean classifier confidence	0.805
Runtime	12.80 s
Throughput	11.88 images/s
These figures describe pipeline behavior, not exact string-recognition accuracy. The test images can contain multiple/background vehicles, which explains why 185 plates were detected in 152 images. Exact plate accuracy requires ground-truth transcription and sequence-level comparison.

Design notes
Character identity is predicted by ResNet18; the character YOLO primarily localizes char and iran regions.
Character crops use zero extra padding because positive padding degraded the existing classifier in prior experiments.
Characters are globally sorted by horizontal center. The first six form the serial section and the rightmost two form the province section.
The iran region is separate and positioned above the province digits; it is not used as a character-order split point.
Province membership is a soft diagnostic rather than a hard rejection rule because the current province list is incomplete.
Limitations
Small, blurred, occluded, overexposed, or distant plates may produce partial readings.
Similar-looking Persian digits and letters remain challenging at low resolution.
Full sequence accuracy has not yet been measured against a transcription ground truth.
Video tracking and multi-frame voting are future extensions; current validation focuses on still images.
Persian text is exported correctly in JSON/CSV. Image overlays use romanized text to avoid OpenCV's limited native Persian shaping support.
Suggested repository layout
text

.
├── configs/
│   └── char_classes.json
├── src/
│   ├── char_classifier.py
│   └── models/
│       └── resnet_18.py
├── checkpoints/             # ignored model files
├── run_alpr.py
├── requirements-alpr.txt
└── README.md
Résumé description
Built an end-to-end Iranian automatic license plate recognition system using YOLO11 and a 34-class ResNet18 classifier. Integrated vehicle and plate localization, character detection, geometric sequence parsing, Persian/romanized output, confidence diagnostics, annotated visualization, and JSON/CSV export. Achieved plate detections in 151 of 152 held-out car-crop images at approximately 11.9 images per second on the test PC.

License
Add the intended source-code license before publishing. Verify the redistribution terms of all datasets and model weights independently.

