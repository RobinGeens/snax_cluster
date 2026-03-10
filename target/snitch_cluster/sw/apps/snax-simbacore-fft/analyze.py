#!/usr/bin/env python3
L1, L2, D = 16, 16, 1

print("=== B vector comparison: Golden vs Hardware ===")
print("K_tile=0 (reals), which flat real indices form each B vector:\n")

for n_eff in range(16):
    # Hardware: R12 reads contiguous bytes from hadamard_reorder
    # reals are stored as [r_0, r_1, ..., r_255]
    # N_eff=n reads bytes [16*n : 16*n+15] = r[16*n .. 16*n+15]
    hw_start = n_eff * 16
    hw_end = hw_start + 15

    # Golden: buildStackedInput_L2 column m, rows 0..15
    # B_golden[l][m] = x2Re(m * L2 + l)  (with D=1, f=0)
    # So column m has flat indices: [m*L2, m*L2+1, ..., m*L2+15]
    golden_start = n_eff * L2
    golden_end = golden_start + 15

    match = "MATCH" if hw_start == golden_start else "MISMATCH"
    print(f"  N_eff={n_eff:2d}: HW r[{hw_start:3d}..{hw_end:3d}], Golden r[{golden_start:3d}..{golden_end:3d}]  <- {match}")

print("\n=== Now compare hadamard_reorder layout vs golden B columns ===")
print("hadamard_reorder stores reals as a flat array.")
print("flattenHadamardReordered separates reals/imags from complex L x D matrix.\n")

# The hadamard output is complex (L x D) = (256 x 1)
# flattenHadamardReordered does:
#   reals = col-major flatten of real part (L x D) -> [r_0, r_1, ..., r_255]
#   imags = col-major flatten of imag part (L x D) -> [i_0, i_1, ..., i_255]
#   output = reals ++ imags
#
# So hadamard_reorder[k] = r_k for k in 0..255, and i_{k-256} for k in 256..511
# The flat index k corresponds to complex element z_k of the L-length sequence
#
# buildStackedInput_L2 creates B[l][m] = x2Re(m * L2 + l) = r_{m*16 + l}
# So golden column m reads: r_{16m}, r_{16m+1}, ..., r_{16m+15}
#
# Hardware N_eff=m reads: hadamard_reorder[16m..16m+15] = r_{16m}..r_{16m+15}
#
# These are THE SAME! So the B input ordering matches.

print("Since L1=L2=16, the golden column m reads r[16m..16m+15]")
print("and the hardware N_eff=m also reads r[16m..16m+15].")
print("The B input is CORRECTLY MATCHED when L1=L2.")
print("\nSo the mismatch must be in the OUTPUT layout or the WEIGHT layout.")

print("\n=== Output layout comparison ===")
print("Golden: flattenCD(out, K_M_N) -> flattenMatrix(out, M=2, N=16, Mu=16, Nu=1, rowMajor=True, rowMajorTile=True)")
print("  Tile order (rowMajor=True): (M=0,N=0), (M=0,N=1), ..., (M=0,N=15), (M=1,N=0), ..., (M=1,N=15)")
print("  Each tile: 16 elements (one column of the output matrix)")
print("")
print("Hardware: ISCore loop K_M_N, S2P packs consecutive outputs:")
print("  For K=1 (final): (M=0,N=0),(M=0,N=1) packed, (M=0,N=2),(M=0,N=3) packed, ...")
print("  Memory: bytes[0:15]=C(0..15,N=0), bytes[16:31]=C(0..15,N=1), ...")
print("  This MATCHES the golden layout.")

print("\n=== Weight layout analysis ===")
print("The issue must be in the A (weight) matrix layout.")
print("\nGolden weight: flattenConvFormat(dftWeightPadded, seqLen=32, dInner=48)")
print("  With convUnroll=4, rowsPerTile=16, colsPerTile=24")
print("  nTilesRow=2, nTilesCol=2")
print("  Tile order: (M=0,K=0), (M=1,K=0), (M=0,K=1), (M=1,K=1)")
print("")
print("ISCore consumption with K_M_N:")
print("  K=0: needs A(M=0,K=0) for all N, then A(M=1,K=0) for all N")
print("  K=1: needs A(M=0,K=1) for all N, then A(M=1,K=1) for all N")
print("  Tile order: (M=0,K=0), (M=1,K=0), (M=0,K=1), (M=1,K=1)")
print("  This MATCHES the convFormat tile order.")

print("\n=== BUT: Does the DFT weight matrix structure match the B input layout? ===")
print("DFT weight2 = getDftMatrix(L2=16, inverse=False, realInput=False)")
print("  This is a (2*L2 x 2*L2) = (32 x 32) block matrix:")
print("  [[Re, -Im], [Im, Re]]")  
print("  But looking at EinfftLib line 49: it uses +Im, not -Im!")
print("")
print("  Rows 0..15, Cols 0..15:   Re(W)")
print("  Rows 0..15, Cols 16..31:  Im(W)  <- should this be -Im(W)?")  
print("  Rows 16..31, Cols 0..15:  Im(W)")
print("  Rows 16..31, Cols 16..31: Re(W)")
print("")
print("The golden matmulKernel computes: W_32x32 x B_32x16")
print("  K_tile=0 (cols 0..15 of W, padded to 24): Re(W) and Im(W) blocks")
print("  K_tile=1 (cols 16..31 of W, padded to 24): Im(W) and Re(W) blocks")
print("")
print("The B matrix has: rows 0..15 = reals, rows 16..31 = imags")
print("So the matmul computes:")
print("  Out[m,n] = sum_k W[m,k] * B[k,n]")
print("  For m in 0..15 (Re output):")
print("    = sum_{k=0..15} Re(W)[m,k]*r[n,k] + sum_{k=0..15} Im(W)[m,k]*i[n,k]")
print("")
print("But the CORRECT complex DFT should be:")
print("  Re(output) = Re(W)*Re(input) - Im(W)*Im(input)")
print("  i.e. the Im part should have a MINUS sign for the real output!")
print("")
print("Looking at getDftMatrix line 49:")
print("  (ib==0, jb==1): W_imag(ii)(jj)  <- this is +Im, but should be -Im for correct DFT")
print("  This is the weight for the real-output rows times imag-input columns")
print("  So the golden computation does Re*Re + Im*Im instead of Re*Re - Im*Im")
print("  >>> This might be intentional (user chose this) or a bug in getDftMatrix <<<")
