from typing import Tuple

import time

import numpy as np
from Pyfhel import Pyfhel, PyCtxt

from perm_llm.common.utils import test_func


class BFV:
    def __init__(self, HE = None):
        if HE is None:
            HE = Pyfhel()           # Creating empty Pyfhel object
            bfv_params = {
                'scheme': 'BFV',    # can also be 'bfv'
                'n': 2**12,         # Polynomial modulus degree, the num. of slots per plaintext,
                                    #  of elements to be encoded in a single ciphertext in a
                                    #  2 by n/2 rectangular matrix (mind this shape for rotations!)
                                    #  Typ. 2^D for D in [10, 16]
                # 't': 65537,         # Plaintext modulus. Encrypted operations happen modulo t
                                    #  Must be prime such that t-1 be divisible by 2^N.
                't_bits': 19,       # Number of bits in t. Used to generate a suitable value
                                    #  for t. Overrides t if specified.
                'sec': 128,         # Security parameter. The equivalent length of AES key in bits.
                                    #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
                                    #  More means more security but also slower computation.
            }
            HE.contextGen(**bfv_params)  # Generate context for bfv scheme
            HE.keyGen()             # Key Generation: generates a pair of public/secret keys
            HE.rotateKeyGen()       # Rotate key generation --> Allows rotation/shifting
            HE.relinKeyGen()        # Relinearization key generation
        self.HE = HE
        self.n_slots = self.HE.get_nSlots()

    def encode_vector(self, vector: np.ndarray):
        n_slots = self.HE.get_nSlots()
        return [self.HE.encodeInt(vector[j: j + n_slots]) for j in range(0, vector.shape[0], n_slots)]

    def encrypt_vector(self, vector: np.ndarray):
        n_slots = self.HE.get_nSlots()
        return [self.HE.encrypt(vector[j: j + n_slots]) for j in range(0, vector.shape[0], n_slots)]

    def decrypt(self, ct: PyCtxt):
        return self.HE.decrypt(ct)

    def cp_dot(self, cts, pts):
        if len(cts) != len(pts):
            raise ValueError(f"The length of the cts and pts must be same, got {len(cts), len(pts)}.")
        return self.HE.cumul_add(sum([(cts[i]*pts[i]) for i in range(len(cts))]))
    
    def serialize(self) -> Tuple[bytes, bytes, bytes]:
        return self.HE.to_bytes_context(), self.HE.to_bytes_public_key(), self.HE.to_bytes_rotate_key()

    def serialize_ciphertext(self, ct: PyCtxt) -> bytes:
        return ct.to_bytes()
    
    def ciphertext_from_bytes(self, ct_bytes: bytes) -> PyCtxt:
        return PyCtxt(pyfhel=self.HE, bytestring=ct_bytes)

    @staticmethod
    def from_bytes(context_bytes: bytes, public_key_bytes: bytes, rotate_key_bytes: bytes):
        HE = Pyfhel()
        HE.from_bytes_context(context_bytes)
        HE.from_bytes_public_key(public_key_bytes)
        HE.from_bytes_rotate_key(rotate_key_bytes)
        return BFV(HE)


if __name__ == "__main__":
    @test_func
    def test_dot():
        length = 130_000
        v1 = np.arange(length)
        v2 = np.zeros(length)
        v2[8964] = 1
        t0 = time.time()

        bfv = BFV()

        pts1 = bfv.encode_vector(v1)
        cts2 = bfv.encrypt_vector(v2)
        prod = bfv.decrypt(bfv.cp_dot(cts2, pts1))[0]
        print(prod)
        print(f"Time consumed: {time.time() - t0:.4f}s")

    @test_func
    def test_serialized_dot():
        length = 130_000
        v1 = np.arange(length)
        v2 = np.zeros(length)
        v2[8964] = 1
        t0 = time.time()

        bfv = BFV()
        bfv_public = BFV.from_bytes(*bfv.serialize())

        pts1 = bfv.encode_vector(v1)
        cts2 = bfv.encrypt_vector(v2)

        cts2 = bfv_public.ciphertext_from_byets(bfv.serialize_ciphertext(cts2))

        ct_prod = bfv.cp_dot(cts2, pts1)
        ct_prod = bfv.ciphertext_from_byets(bfv_public.serialize_ciphertext(ct_prod))

        print(bfv.decrypt(ct_prod)[0])
        print(f"Time consumed: {time.time() - t0:.4f}s")

    test_dot()