from dorian import raytrace
import tracemalloc, sys
import os
nthreads=40
os.environ["OMP_NUM_THREADS"] = str(nthreads)

if __name__ == "__main__":
    tracemalloc.start()
    raytrace(
        simDir=sys.argv[1],
        outDir=sys.argv[2],
        z_s=float(sys.argv[3]),
        interp=sys.argv[4],
        max_time_in_sec=int(sys.argv[5]),
        restart_file="" if len(sys.argv) == 6 else sys.argv[6],
        save_ray_positions=False,
        nthreads=nthreads,
    )

    current, peak = tracemalloc.get_traced_memory()

    print(
        f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB; Diff = {(peak - current) / 10**6}MB",
        flush=True,
    )

    tracemalloc.stop()
