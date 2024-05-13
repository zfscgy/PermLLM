import time

import numpy as np
from Pyfhel import Pyfhel

HE = Pyfhel()           # Creating empty Pyfhel object
bfv_params = {
    'scheme': 'BFV',    # can also be 'bfv'
    'n': 2**12,         # Polynomial modulus degree, the num. of slots per plaintext,
                        #  of elements to be encoded in a single ciphertext in a
                        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
                        #  Typ. 2^D for D in [10, 16]
    # 't': 65537,         # Plaintext modulus. Encrypted operations happen modulo t
                        #  Must be prime such that t-1 be divisible by 2^N.
    't_bits': 18,       # Number of bits in t. Used to generate a suitable value
                        #  for t. Overrides t if specified.
    'sec': 128,         # Security parameter. The equivalent length of AES key in bits.
                        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
                        #  More means more security but also slower computation.
}
HE.contextGen(**bfv_params)  # Generate context for bfv scheme
HE.keyGen()             # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()       # Rotate key generation --> Allows rotation/shifting
HE.relinKeyGen()        # Relinearization key generation

print("\n1. Pyfhel FHE context generation")
print(f"\t{HE}")



length = 130_000
v1 = np.arange(length)
v2 = np.zeros(length)
v2[8964] = 1


t0 = time.time()
pt1 = [HE.encodeInt(v1[j:j+HE.get_nSlots()]) for j in range(0, length, HE.get_nSlots())]
ct2 = [HE.encrypt(v2[j:j+HE.get_nSlots()]) for j in range(0, length, HE.get_nSlots())]

cRes = sum([~(ct2[i]*pt1[i]) for i in range(len(ct2))])
cRes = HE.cumul_add(cRes)
print(HE.decryptInt(cRes))

print(f"Time consumed: {time.time() - t0:.4f}s")