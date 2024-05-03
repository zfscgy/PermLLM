from typing import Union, List

import numpy as np
import tenseal as ts
import tenseal.sealapi as seal

import threading

from split_llm.common.utils import test_func


class BFV:
    def __init__(self, context: ts.Context = None) -> None:
        """
        BFV cryptosystem
        https://github.com/OpenMined/TenSEAL/blob/main/tutorials%2FTutorial%200%20-%20Getting%20Started.ipynb
        """
        self.ciphertext_size = 2048
        if context is None:
            self.context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=4096, 
                    plain_modulus=1032193,
                    coeff_mod_bit_sizes=[36,36,36]
                )
            self.context.generate_galois_keys()
        else:
            self.context = context

    def enc_vector(self, vec: np.ndarray):
        return ts.bfv_vector(self.context, vec)

    def decrypt(self, ct):
        if isinstance(ct, ts.BFVTensor):
            return np.array(ct.decrypt().tolist())
        elif isinstance(ct, ts.BFVVector):
            return np.array(ct.decrypt())
        else:
            raise TypeError("Not a valid ciphertext.")
    
    def serialize(self):
        """
        The private key will not be serialized!
        """
        return self.context.serialize()
    
    @staticmethod
    def from_bytes(serialized: bytes):
        return BFV(context=ts.context_from(serialized))
        

    
if __name__ == "__main__":
    
    @test_func
    def test_BFV():
        bfv = BFV()
        v1 = np.arange(130000)
        v2 = np.zeros(130000)
        v2[-1] = 1
        step_size = 2048
        c_dot_results = [None for _ in range(np.ceil(130000 / step_size).astype(int))]

        def enc_dot(i):
            c2 = bfv.enc_vector(v2[i: i + step_size])
            d = c2.dot(v1[i:i + step_size])
            c_dot_results[i // 2048] = d

        # for i in range(0, 130000, step_size):
        #     c_dot_threads.append(threading.Thread(target=enc_dot, args=(i,)))

        # for t in c_dot_threads:
        #     t.start()
        
        # for t in c_dot_threads:
        #     t.join()
        
        for i in range(0, 130000, step_size):
            enc_dot(i)


        d = c_dot_results[0]
        for dd in c_dot_results[1:]:
            d = d + dd

        np.set_printoptions(precision=4, suppress=True)
        print(np.array(bfv.decrypt(d)))

    @test_func
    def test_serialize():
        bfv = BFV()
        
        bfv2 = BFV.from_bytes(bfv.serialize())

        ct = bfv2.enc_vector([1926, 817])
        serialized_ct = ct.serialize()
        
        restored_ct = ts.bfv_vector_from(bfv.context, serialized_ct)
        recovered_pt = bfv.decrypt(restored_ct)
        print(recovered_pt)


    test_serialize()