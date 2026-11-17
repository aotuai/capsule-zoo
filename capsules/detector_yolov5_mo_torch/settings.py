# True: the model's IR divides the image by 255 and swaps BGR to RGB itself (the usual OpenVINO conversion).
# False: the capsule does it. Set this to match the model you add.
model_includes_preprocessing = True
