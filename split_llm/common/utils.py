import time


def test_func(func): 
    def wrap(*args, **kwargs): 
        print(f"\n===========\nStart testing {func.__name__}\n================") 

        start = time.time() 
        result = func(*args, **kwargs) 
        end = time.time()

        print(f"Test {func.__name__}, finished in {end-start:.4f}s") 
        return result 
    return wrap