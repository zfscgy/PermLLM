from typing import Union, List

# from queue import Queue

import time

import numpy as np
import tenseal as ts
import tenseal.sealapi as seal

import threading
import multiprocessing

from perm_llm.common.utils import test_func


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
                    coeff_mod_bit_sizes=[36,36,36],
                    n_threads=8
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
        length = 1000_000
        v1 = np.arange(length)
        v2 = np.zeros(length)
        v2[8964] = 1
        step_size = 2048
        n_cts = np.ceil(length / step_size).astype(int)
        c_dot_results = multiprocessing.Queue()

        def enc_dot(v1_segment, v2_segment):
            c2 = bfv.enc_vector(v2_segment)
            d = c2.dot(v1_segment)
            return d

        def enc_dot_range(v1_segments, v2_segments, start: int, end: int, results):
            t0 = time.time()
            print(f"Task {start}:{end} starts")
            ds = []
            for i in range(start, end):
                d = enc_dot(v1_segments[i - start], v2_segments[i - start])
                ds.append(d)
            
            for d in ds:
                results.put(d)
        
            print(f"Task {start}:{end} ends, time consumed {time.time() - t0:.4f}s")

        t0 = time.time()

        c_dot_threads = []

        n_threads = 1
        n_segs_per_thread = int(np.ceil(n_cts / n_threads))

        for i in range(n_threads):
            v1_segments = []
            v2_segments = []
            start_seg = i * n_segs_per_thread
            end_seg = min(i * n_segs_per_thread + n_segs_per_thread, n_cts)
            
            for j in range(start_seg, end_seg):
                v1_segments.append(v1[j * step_size: j * step_size + step_size])
                v2_segments.append(v2[j * step_size: j * step_size + step_size])
    
            c_dot_threads.append(multiprocessing.Process(target=enc_dot_range, args=(v1_segments, v2_segments, start_seg, end_seg, c_dot_results)))

        for t in c_dot_threads:
            t.start()
        
        for t in c_dot_threads:
            t.join()

        # for i in range(n_cts):
        #     enc_dot(v1[i * step_size: i * step_size + step_size], v2[i * step_size: i * step_size + step_size], c_dot_results)

        print(f"Dot product time: {time.time() - t0:.4f}s")

        t0 = time.time()
        d = c_dot_results.get()
        for dd in c_dot_results.queue:
            d = d + dd

        print(f"Sum time: {time.time() - t0:.4f}s")

        t0 = time.time()
        np.set_printoptions(precision=4, suppress=True)
        print(np.array(bfv.decrypt(d)))
        print(f"Decryption time: {time.time() - t0:.4f}s")


    @test_func
    def test_BFV_onect():
        bfv = BFV()
        v1 = np.arange(130000)
        v2 = np.zeros(130000)

        t0 = time.time()
        c2 = bfv.enc_vector(v2)
        d = c2.dot(v1)

        print(f"Sum time: {time.time() - t0:.4f}s")

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


    test_BFV()