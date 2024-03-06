from typing import Union, List

import numpy as np
import tenseal as ts
import tenseal.sealapi as seal


class CKKS:
    def __init__(self, modular_coeff_bits: List[int] = None, poly_degree: int = None) -> None:
        """
        modular_coeff_bits: [p0, p1, ..., pn], where at i-th multiplication level, the modulus is p0 * p1 * ... * pi. 
        Thus, p1, ..., pi should be approximate to the scale factor, since during each multpplication, the scale factor will be multiplied. (as * bs = abs^2)
        The first modulus has to store both the integer part and the precision part, so it shall be larger.
        The last modulus is special, however, I still do not understand its role in CKKS. Conventionally, it is chosen the same value as the first modulus.
        More information can be found in https://blog.openmined.org/ckks-explained-part-5-rescaling/
        """
        self.modular_coeff_bits = modular_coeff_bits or [40, 20, 40]  # Only for one multiplication
        self.poly_degree = poly_degree or 4096
        self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_degree,
                coeff_mod_bit_sizes=self.modular_coeff_bits
            )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**20


    def enc_vector(self, vec: np.ndarray):
        return ts.ckks_vector(self.context, vec)

    def decrypt(self, ct):
        if isinstance(ct, ts.CKKSTensor):
            return np.array(ct.decrypt().tolist())
        elif isinstance(ct, ts.CKKSVector):
            return np.array(ct.decrypt())
        else:
            raise TypeError("Not a valid ciphertext.")

    
if __name__ == "__main__":
    ckks = CKKS()
    v1 = np.arange(2048  )
    v2 = np.array([1] * 2048)
    c1 = ckks.enc_vector(v1)
    for i in range(10):
        c2 = ckks.enc_vector(v2)
        d = c1.mul(c2)
        d = d.decrypt()
    np.set_printoptions(precision=4, suppress=True)
    print(np.array(d))