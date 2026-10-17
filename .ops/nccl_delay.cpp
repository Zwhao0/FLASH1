// Experiment-only link delay. Loaded by Greyhound's NCCL_PATH; no host network edits.
#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <time.h>
#include <cerrno>
static void *real_nccl() {
    static void *h = dlopen("/usr/lib/x86_64-linux-gnu/libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
    return h;
}
static void CUDART_CB pause_stream(void *p) {
    long us = reinterpret_cast<long>(p);
    timespec t{us / 1000000, (us % 1000000) * 1000};
    while (nanosleep(&t, &t) && errno == EINTR) {}
}
static void add_delay(size_t count, int peer, ncclComm_t comm, cudaStream_t stream) {
    const char *fpath = getenv("GREYHOUND_DELAY_FILE");
    if (!fpath || count < 1024) return;
    int rank = -1, size = 0;
    using query_t = ncclResult_t (*)(ncclComm_t, int*);
    reinterpret_cast<query_t>(dlsym(real_nccl(), "ncclCommUserRank"))(comm, &rank);
    reinterpret_cast<query_t>(dlsym(real_nccl(), "ncclCommCount"))(comm, &size);
    if (size != 4 || !((rank == 1 && peer == 2) || (rank == 2 && peer == 1))) return;
    FILE *f = fopen(fpath, "r");
    if (!f) return;
    double ms = 0;
    int parsed = fscanf(f, "%lf", &ms); fclose(f);
    if (parsed == 1 && ms > 0 && ms <= 100)
        cudaLaunchHostFunc(stream, pause_stream, reinterpret_cast<void*>(static_cast<long>(ms * 1000)));
}
extern "C" ncclResult_t ncclSend(const void *b, size_t n, ncclDataType_t d, int p, ncclComm_t c, cudaStream_t s) {
    add_delay(n, p, c, s);
    using fn = ncclResult_t (*)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
    return reinterpret_cast<fn>(dlsym(real_nccl(), "ncclSend"))(b,n,d,p,c,s);
}
extern "C" ncclResult_t ncclRecv(void *b, size_t n, ncclDataType_t d, int p, ncclComm_t c, cudaStream_t s) {
    add_delay(n, p, c, s);
    using fn = ncclResult_t (*)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
    return reinterpret_cast<fn>(dlsym(real_nccl(), "ncclRecv"))(b,n,d,p,c,s);
}
