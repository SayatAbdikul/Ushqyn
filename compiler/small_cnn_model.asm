; Custom Architecture Assembly Code
; Generated from ONNX model

LOAD_V 9, 0xc0, 784
LOAD_M 1, 0x3000, 4, 9
LOAD_V 3, 0x4c0, 4
CONV2D_CFG 10, 28, 28, 1, 4, 3, 3, 1, 0
CONV2D_RUN 10, 9, 1, 3, 0
MAXPOOL 12, 10, 26, 26, 4, 2, 2
LOAD_M 2, 0x3080, 8, 36
LOAD_V 4, 0x4c4, 8
CONV2D_CFG 11, 13, 13, 4, 8, 3, 3, 1, 0
CONV2D_RUN 11, 12, 2, 4, 1
MAXPOOL 13, 11, 11, 11, 8, 2, 2
LOAD_M 1, 0x940, 10, 224
LOAD_V 3, 0x4cc, 10
GEMV 5, 1, 13, 3, 10, 200
STORE 5, 0x8c0, 10