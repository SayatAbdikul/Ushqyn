; Custom Architecture Assembly Code
; Generated from ONNX model

LOAD_V 9, 0xc0, 784
LOAD_M 16, 0x3000, 4, 9
LOAD_V 0, 0x4c0, 4
CONV2D_CFG 1, 28, 28, 1, 4, 3, 3, 1, 0
CONV2D_RUN 1, 9, 16, 0, 1
MAXPOOL 2, 1, 26, 26, 4, 2, 2
LOAD_M 16, 0x3080, 8, 36
LOAD_V 1, 0x4c4, 8
CONV2D_CFG 3, 13, 13, 4, 8, 3, 3, 1, 0
CONV2D_RUN 3, 2, 16, 1, 1
MAXPOOL 1, 3, 11, 11, 8, 2, 2
LOAD_M 16, 0x940, 10, 224
LOAD_V 0, 0x4cc, 10
GEMV 2, 16, 1, 0, 10, 200
STORE 2, 0x8c0, 10