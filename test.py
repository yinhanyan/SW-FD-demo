# first clone the Sliding-Window-FD
# (https://github.com/yinhanyan/Sliding-Window-FD)
# git clone https://github.com/yinhanyan/Sliding-Window-FD.git swfd --depth 1

from swfd import SeqBasedSWFD
# or `from swfd import *`
import numpy as np

if __name__ == "__main__":
    l = 16
    N = 64
    d = 32
    R = 2**16 - 1
    epochs = 2048
    sw_fd = SeqBasedSWFD(N, R, d, l)

    random_vector = np.random.randn(1, d)
    a = np.sqrt(R) * random_vector / np.linalg.norm(random_vector, ord=2)
    sw_fd.fit(a)
    B_t, _, _, delta = sw_fd.get()
    print(B_t)
