import subprocess

import numpy as np
import matplotlib.pyplot as plt


command = "examples/layer_norm_bench/layer_norm_bench {} {}"

M = [1024, 4096, 8192, 16348, 32768]
N = [128, 320, 640, 1022, 2048, 5000, 8192, 16348, 24004, 32772, 50000]


def run(M, N):
    out = subprocess.run(["examples/layer_norm_bench/layer_norm_bench", str(M), str(N)], capture_output=True)
    output = str(out.stdout.decode())
    elapsed = output.rstrip().split(",")
    return [float(t) for t in elapsed]


for m in M:
    cutlass = []
    simple_half8 = []
    layer_norm_smem = []
    layer_norm_smem_async = []

    ns = []
    for n in N:
        if m == 32768 and n == 50000:
            continue

        ns.append(n)

        t1, t2, t3, t4 = run(m, n)
        cutlass.append(t1)
        simple_half8.append(t2)
        layer_norm_smem.append(t3)
        layer_norm_smem_async.append(t4)

    plt.plot(ns, cutlass, label="cutlass")
    plt.plot(ns, simple_half8, label="simple half8 kernel")
    plt.plot(ns, layer_norm_smem, label="smem kernel")
    plt.plot(ns, layer_norm_smem_async, label="smem + async kernel")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("time in us")
    plt.title("M = {}".format(m))
    plt.savefig("M_{}.png".format(m))
