#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/perf/IndexWrapper.h>

std::map<int, double> evaluate2(
    faiss::Index& index,
    const float* xq,
    const faiss::idx_t* gt,
    int k,
    size_t nq, 
    double& elpsTime) {

    auto start_time = std::chrono::steady_clock::now();

    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    index.search(nq, xq, k, D.data(), I.data());

    auto end_time = std::chrono::steady_clock::now();

    std::map<int, double> recalls;

    int i = 1;
    while (i <= k) {
        int count = 0;
        for (size_t j = 0; j < nq; ++j) {
            std::vector<faiss::idx_t> I_segment(I.begin() + j * k, I.begin() + j * k + i);
            std::vector<faiss::idx_t> gt_segment(gt + j, gt + j + 1);

            if (I_segment[j] == gt_segment[j]) {
                count++;
            }
        }
        recalls[i] = static_cast<double>(count) / static_cast<double>(nq);
        i *= 10;
    }

    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double time_per_query = elapsed_time / nq;
    elpsTime = time_per_query;
    return recalls;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {

    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {    
    
    double t0 = elapsed();

    std::cout << "load data" << std::endl;

    float* xq;
    float* xb;
    float* xt;
    size_t nq;
    size_t d;
    size_t nb, d2;
    size_t nt, k;
    faiss::idx_t* gt;
    
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);
        xq = fvecs_read("sift1M/sift_query.fvecs", &d, &nq);
    
        printf("[%.3f s] Loading database\n", elapsed() - t0);
        xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);

        assert(d == d2 || !"query does not have the same dimension as the train set");

        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);
        
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        size_t nq2;
        int* gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;

    }
       
    std::cout << "============ Exact search" << std::endl;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig flat_config;
    flat_config.device = 0;
    faiss::gpu::GpuIndexFlatL2 index(&res, d, flat_config);
    std::cout << "add vectors to index" << std::endl;
    index.add(nq, xb);

    std::cout << "warmup" << std::endl;
    std::vector<float> distance(nq * 123, 0);
    std::vector<faiss::idx_t> indices(nq * 123, 0);
    index.search(nq, xq, 123, distance.data(), indices.data());

    for (auto d : distance) {
        assert(d == std::numeric_limits<float>::max() || "distances overflow!!");
    }

    for (auto i : indices) {
        assert(i == -1 || "indices overflow!!");
    }

    std::cout << "benchmark" << std::endl;
    for (int lk = 0; lk < 11; lk++) {
        int k = 1 << lk;
        double elpsTime;
        auto recalls = evaluate2(index, xq, gt, k, nq, elpsTime);
        std::cout << "k=" << k << "\t" << elpsTime << "ms\t" << "R@1" << ": " << recalls[1] << "\tR@10" << ": " << recalls[10]  << "\tR@100" << ": " << recalls[100] << "\n";
    }
    std::cout << "\n";
    std::cout << "============ Approximate search" << std::endl;
    faiss::Index* index_ivf = faiss::index_factory(d, "IVF4096,PQ64");
    faiss::gpu::GpuClonerOptions co;
    co.useFloat16 = true;

    faiss::gpu::GpuResourcesProvider* resProvider = new faiss::gpu::StandardGpuResources();
    faiss::Index* index_gpu = faiss::gpu::index_cpu_to_gpu(resProvider, 0, index_ivf);


    std::cout << "train" << std::endl;
    index_gpu->train(nq, xt);
    std::cout << "add vectors to index" << std::endl;
    index_gpu->add(nq, xb);

    std::cout << "warmup" << std::endl;
    index_gpu->search(nq, xq, 123, distance.data(), indices.data());

    std::cout << "benchmark" << std::endl;    
    for (int lk = 0; lk < 10; lk++) {
        int k = 1 << lk;
        double elpsTime;
        auto recalls = evaluate2(*index_gpu, xq, gt, k, nq, elpsTime);
        std::cout << "nprobe= " << k << "\t" << elpsTime << "\tms,\t" << "recalls=" << recalls[1] << "\t" << recalls[10]  << "\t" << recalls[100] << "\n";
    }

    delete index_ivf;

    return 0;
}