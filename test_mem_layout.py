dState = 64
FP8 = 8
suc_serial_width_BC = 2 * 64
seqLenUnroll = 16
seqLen = 96
dInner = 2 * 96
delaySU = 4
xProjDim = 24
BANK_BYTES = 8


def addr_to_bank(addr):
    return (addr // BANK_BYTES) % 32


bounds = [
    (2 * dState * FP8) // (2 * suc_serial_width_BC),  #
    seqLenUnroll,
    seqLen // seqLenUnroll,
    dInner // delaySU,  # Irrelevant dimension
]
strides = [
    (2 * suc_serial_width_BC // 8) * seqLenUnroll,
    BANK_BYTES,
    seqLenUnroll * xProjDim * FP8 // 8,
    0,
]
SS = (seqLenUnroll + 1) * BANK_BYTES  # Spatial stride


for i in range(bounds[-1]):
    for j in range(bounds[-2]):
        for k in range(bounds[-3]):
            for l in range(bounds[-4]):
                base_addr = l * strides[0] + k * strides[1] + j * strides[2] + i * strides[3]
                spatial_addr = [base_addr + s * SS for s in range(4)]
                banks = [addr_to_bank(addr) for addr in spatial_addr]
                print(f"{spatial_addr} -> {banks}")
