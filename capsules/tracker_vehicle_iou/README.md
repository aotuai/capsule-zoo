# About the Capsule
## Usage
This capsule is for adding tracks to vehicles. It should be used in conjunciton
with a vehicle detector.

## Methodology
This capsule uses Intersection Over Union based tracking to match vehicles
from one frame to the next. The tracking algorithm can be found under tracker.py

It is based on the algorithm from this paper:  
[High-Speed Tracking-by-Detection Without Using Image Information](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf).

It is very fast, and typically very accurate. 
