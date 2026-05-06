#include "Vmaxpool_execution_tb_wrapper.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <verilated.h>

#define TILE_ELEMS 32

void tick(Vmaxpool_execution_tb_wrapper *top) {
  top->clk = 0;
  top->eval();
  top->clk = 1;
  top->eval();
}

void preload_vector(Vmaxpool_execution_tb_wrapper *top, int buf_id,
                    int num_elements, const std::vector<int8_t> &data) {
  top->preload_en = 1;
  top->preload_buf_id = buf_id;

  int cycles = (num_elements + TILE_ELEMS - 1) / TILE_ELEMS;

  for (int c = 0; c < cycles; c++) {
    for (int i = 0; i < TILE_ELEMS; i++) {
      int idx = c * TILE_ELEMS + i;
      top->preload_data[i] = (idx < data.size()) ? data[idx] : 0;
    }
    tick(top);
  }
  top->preload_en = 0;
  tick(top); // margin
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Vmaxpool_execution_tb_wrapper *top = new Vmaxpool_execution_tb_wrapper;

  top->clk = 0;
  top->rst = 1;
  top->start = 0;
  top->preload_en = 0;
  top->preload_buf_id = 0;
  for (int i = 0; i < TILE_ELEMS; i++)
    top->preload_data[i] = 0;

  // Reset sequence
  for (int i = 0; i < 5; i++)
    tick(top);
  top->rst = 0;
  tick(top);

  // Test parameters
  // Input: 1 channel, 4x4
  top->fmap_h = 4;
  top->fmap_w = 4;
  top->in_channels = 1;
  top->pool_size = 2;
  top->stride_val = 2;

  std::vector<int8_t> x_data = {1,  4,  0, 2, 3, 9, 5, 1,
                                -5, -2, 7, 8, 0, 1, 6, 3};
  // Expected output:
  // Window1: max(1,4,3,9) = 9
  // Window2: max(0,2,5,1) = 5
  // Window3: max(-5,-2,0,1) = 1
  // Window4: max(7,8,6,3) = 8
  // Result: [9, 5, 1, 8]

  int x_buf_id = 1;
  int out_buf_id = 2;

  std::cout << "Preloading buffers for MaxPool..." << std::endl;
  preload_vector(top, x_buf_id, x_data.size(), x_data);

  top->x_buffer_id = x_buf_id;
  top->dest_buffer_id = out_buf_id;

  std::cout << "Starting MaxPool Evaluation..." << std::endl;
  top->start = 1;
  tick(top);
  top->start = 0;

  int timeout = 5000;
  while (!top->done && timeout > 0) {
    tick(top);
    timeout--;
  }

  if (timeout <= 0) {
    std::cerr << "TIMEOUT: MaxPool evaluation failed to hit DONE signal."
              << std::endl;
    delete top;
    return 1;
  }

  std::cout << "MaxPool Evaluation Completed Successfully!" << std::endl;

  delete top;
  return 0;
}
