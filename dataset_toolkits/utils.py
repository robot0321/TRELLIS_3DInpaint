from typing import *
import hashlib
import numpy as np


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def upper_hemisphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)

    u = (u + offset[0] / num_samples) % 1.0
    v = (v + offset[1]) % 1.0

    # 핵심 변경
    theta = np.arccos(u)        # [0, pi/2]
    phi = 2 * np.pi * v

    return [phi, theta]

def hemisphere_fibonacci(i, N):
    # z를 균등(면적 균등)하게
    z = (i + 0.5) / N              # in (0,1)
    theta = np.arccos(z)

    # golden angle로 phi를 배치 (규칙적이지만 “덜 격자”)
    golden = (1 + 5**0.5) / 2
    phi = 2 * np.pi * ((i / golden) % 1.0)
    return phi, theta