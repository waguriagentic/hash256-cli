#!/usr/bin/env python3
"""
GPU-accelerated keccak256 miner using CuPy CUDA kernels.
Optimized for NVIDIA GPUs (RTX series and above).

Usage: python3 gpu-keccak-miner.py <challenge_hex> <difficulty_hex>
"""

import sys
import json
import time
import struct
import numpy as np

def log(msg):
    print(f"[GPU] {msg}", file=sys.stderr, flush=True)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    log("CuPy imported successfully")
except ImportError as e:
    GPU_AVAILABLE = False
    log(f"CuPy import failed: {e}")


# ─── Keccak-256 CUDA Kernel ───────────────────────────────
# Full keccak-256 implementation in CUDA C
KECCAK_CUDA_SOURCE = r'''
extern "C" {

// Keccak round constants
__constant__ unsigned long long RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int ROTC[24] = {
     1,  3,  6, 10, 15, 21, 28, 36,
    45, 55,  2, 14, 27, 41, 56,  8,
    25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int PILN[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,
     8, 21, 24,  4, 15, 23, 19, 13,
    12,  2, 20, 14, 22,  9,  6,  1
};

__device__ unsigned long long rotl64(unsigned long long x, int y) {
    return (x << y) | (x >> (64 - y));
}

__device__ void keccak_f1600(unsigned long long* state) {
    unsigned long long bc[5];
    
    for (int round = 0; round < 24; round++) {
        // Theta
        for (int i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        
        for (int i = 0; i < 5; i++) {
            unsigned long long t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                state[j + i] ^= t;
        }
        
        // Rho Pi
        unsigned long long t = state[1];
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            bc[0] = state[j];
            state[j] = rotl64(t, ROTC[i]);
            t = bc[0];
        }
        
        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++)
                bc[i] = state[j + i];
            for (int i = 0; i < 5; i++)
                state[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        
        // Iota
        state[0] ^= RC[round];
    }
}

__device__ void keccak256(const unsigned char* input, int len, unsigned char* output) {
    unsigned long long state[25];
    
    // Init state
    for (int i = 0; i < 25; i++) state[i] = 0;
    
    // Absorb
    int rate = 136; // 1088/8 = 136 bytes (keccak256 rate)
    int blocks = (len + rate - 1) / rate;
    
    for (int b = 0; b < blocks; b++) {
        int offset = b * rate;
        int block_len = min(rate, len - offset);
        
        for (int i = 0; i < block_len; i += 8) {
            unsigned long long v = 0;
            for (int j = 0; j < 8 && (i + j) < block_len; j++)
                v |= (unsigned long long)input[offset + i + j] << (j * 8);
            state[i / 8] ^= v;
        }
        
        keccak_f1600(state);
    }
    
    // Padding
    int offset = (blocks - 1) * rate;
    int last_len = len - offset;
    
    // Pad with 0x06 (keccak256 domain separator) + 0x80
    unsigned char pad[136];
    for (int i = 0; i < 136; i++) pad[i] = 0;
    for (int i = 0; i < last_len; i++) pad[i] = input[offset + i];
    pad[last_len] = 0x06;
    pad[rate - 1] |= 0x80;
    
    for (int i = 0; i < rate; i += 8) {
        unsigned long long v = 0;
        for (int j = 0; j < 8; j++)
            v |= (unsigned long long)pad[i + j] << (j * 8);
        state[i / 8] ^= v;
    }
    
    keccak_f1600(state);
    
    // Squeeze (first 32 bytes)
    for (int i = 0; i < 4; i++) {
        unsigned long long v = state[i];
        for (int j = 0; j < 8; j++)
            output[i * 8 + j] = (v >> (j * 8)) & 0xFF;
    }
}

__global__ void keccak_mine_kernel(
    const unsigned char* challenge,  // 32 bytes
    const unsigned char* difficulty, // 32 bytes (big-endian)
    unsigned long long start_nonce,
    unsigned long long* result_nonce, // output: found nonce (0 = not found)
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    unsigned long long nonce = start_nonce + (unsigned long long)idx;
    
    // Build input: 32 bytes challenge + 32 bytes nonce (big-endian)
    unsigned char input[64];
    for (int i = 0; i < 32; i++) input[i] = challenge[i];
    
    // Write nonce big-endian
    for (int i = 63; i >= 32; i--) {
        input[i] = nonce & 0xFF;
        nonce >>= 8;
    }
    
    // Compute keccak256
    unsigned char hash[32];
    keccak256(input, 64, hash);
    
    // Compare hash < difficulty (big-endian comparison)
    bool less = false;
    for (int i = 0; i < 32; i++) {
        if (hash[i] < difficulty[i]) { less = true; break; }
        if (hash[i] > difficulty[i]) { break; }
    }
    
    if (less) {
        // Found! Use atomic to store first found nonce
        atomicMin(result_nonce, start_nonce + (unsigned long long)idx);
    }
}

} // extern "C"
'''


def compile_kernel():
    """Compile the CUDA kernel."""
    log("Compiling CUDA kernel...")
    try:
        module = cp.RawModule(code=KECCAK_CUDA_SOURCE)
        kernel = module.get_function('keccak_mine_kernel')
        log("CUDA kernel compiled successfully")
        return kernel
    except Exception as e:
        log(f"CUDA kernel compilation failed: {e}")
        raise


def gpu_mine(challenge_hex: str, difficulty_hex: str, timeout_sec: int = 60):
    """Run GPU mining."""
    if not GPU_AVAILABLE:
        log("GPU not available (CuPy not installed)")
        return {"found": False, "error": "CuPy not available"}
    
    # Parse inputs
    challenge = bytes.fromhex(challenge_hex[2:] if challenge_hex.startswith("0x") else challenge_hex)
    diff_hex = difficulty_hex[2:] if difficulty_hex.startswith("0x") else difficulty_hex
    difficulty = bytes.fromhex(diff_hex.zfill(64))
    
    log(f"Challenge: {challenge_hex[:20]}...")
    log(f"Difficulty: {difficulty_hex[:20]}...")
    
    # Compile kernel
    try:
        kernel = compile_kernel()
    except Exception as e:
        return {"found": False, "error": f"CUDA compile failed: {str(e)}"}
    
    # GPU memory
    log("Allocating GPU memory...")
    d_challenge = cp.asarray(np.frombuffer(challenge, dtype=np.uint8))
    d_difficulty = cp.asarray(np.frombuffer(difficulty, dtype=np.uint8))
    
    # Mining parameters
    BLOCK_SIZE = 256
    GRID_SIZE = 4096  # 4096 blocks * 256 threads = 1M hashes per batch
    BATCH_SIZE = BLOCK_SIZE * GRID_SIZE
    
    log(f"GPU mining config: {GRID_SIZE} blocks x {BLOCK_SIZE} threads = {BATCH_SIZE:,} hashes/batch")
    
    import random
    start_nonce = random.randint(0, 2**40)
    
    start_time = time.time()
    total_hashes = 0
    
    log("Starting GPU mining loop...")
    
    while time.time() - start_time < timeout_sec:
        d_result = cp.zeros(1, dtype=cp.uint64)
        d_result[0] = 0xFFFFFFFFFFFFFFFF  # Max value = not found
        
        kernel(
            (GRID_SIZE,), (BLOCK_SIZE,),
            (d_challenge, d_difficulty, 
             cp.uint64(start_nonce), d_result, cp.int32(BATCH_SIZE))
        )
        
        cp.cuda.Stream.null.synchronize()
        
        result_nonce = int(d_result[0])
        total_hashes += BATCH_SIZE
        
        if result_nonce < 0xFFFFFFFFFFFFFFFF:
            # Found!
            elapsed = time.time() - start_time
            rate = total_hashes / elapsed if elapsed > 0 else 0
            
            log(f"FOUND nonce: {result_nonce}")
            
            return {
                "found": True,
                "nonce": str(result_nonce),
                "hash": "",  # Will be computed by caller
                "hashes": total_hashes,
                "elapsed": elapsed,
                "rate": int(rate)
            }
        
        start_nonce += BATCH_SIZE
        
        # Print progress
        elapsed = time.time() - start_time
        if elapsed > 0 and int(elapsed) % 5 == 0:
            rate = total_hashes / elapsed
            log(f"{total_hashes:,} hashes | {rate/1e6:.1f}M h/s | {elapsed:.0f}s")
    
    elapsed = time.time() - start_time
    rate = total_hashes / elapsed if elapsed > 0 else 0
    log(f"Mining timeout after {elapsed:.0f}s, checked {total_hashes:,} hashes")
    return {
        "found": False,
        "checked": str(total_hashes),
        "elapsed": elapsed,
        "rate": int(rate)
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gpu-keccak-miner.py <challenge_hex> <difficulty_hex>")
        sys.exit(1)
    
    challenge_hex = sys.argv[1]
    difficulty_hex = sys.argv[2]
    
    log("=== GPU Keccak256 Miner ===")
    
    if not GPU_AVAILABLE:
        log("ERROR: CuPy not installed")
        print(json.dumps({"found": False, "error": "CuPy not installed. Install: pip install cupy-cuda12x"}))
        sys.exit(1)
    
    # Check GPU
    try:
        dev = cp.cuda.Device()
        log(f"GPU Device: {dev}")
        mem_info = dev.mem_info
        log(f"GPU Memory: {mem_info[1] / 1024 / 1024:.0f}MB total, {mem_info[0] / 1024 / 1024:.0f}MB free")
    except Exception as e:
        log(f"GPU info error: {e}")
    
    result = gpu_mine(challenge_hex, difficulty_hex)
    
    # Output result as JSON to stdout
    print(json.dumps(result))
    sys.exit(0 if result.get("found") else 1)


if __name__ == "__main__":
    main()
