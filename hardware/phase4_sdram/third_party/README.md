`nestang_sdram.v` is copied without RTL changes from
[`nand2mario/sdram-tang-nano-20k`](https://github.com/nand2mario/sdram-tang-nano-20k),
commit `918ae4143eed676d29b706df6ec7ebcb61e257c1`. It is Apache-2.0;
the original license is included here. This controller accesses one byte
per command and is used for physical SDRAM integrity testing. It does not
provide the burst transfers or throughput required for the final accelerator.
