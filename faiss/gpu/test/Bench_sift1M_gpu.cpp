#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
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

template<typename T>
void vecs_read(const char* fname, size_t& d_out, size_t& n_out, std::vector<T>& data) {
    std::ifstream file(fname, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << fname << std::endl;
        return; 
    }

    size_t d = 0;
    file.read(reinterpret_cast<char*>(&d), sizeof(T));
    d_out = d;

    // Calculate the number of integers in the file (excluding the first integer)
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    size_t numElements = fileSize / sizeof(T);
    size_t numVectors = numElements / (d + 1);
    n_out = numVectors;

    file.seekg(0, std::ios::beg);

    // Read and store the vectors in a 2D vector
    std::vector<std::vector<T>> outVec;
    for (size_t i = 0; i < numVectors; ++i) {
        std::vector<T> vectorData(d + 1);
        file.read(reinterpret_cast<char*>(vectorData.data()), ( d + 1 ) * sizeof(T));
        std::vector<T> subset(vectorData.begin() + 1, vectorData.end());
        outVec.emplace_back(subset);
        file.ignore((d + 1) * sizeof(T));
    }

    file.close();

    std::vector<T> result(d * numVectors);
    size_t currentIndex = 0;

    // Copy elements to the 1D array
    for (const auto& innerVec : outVec) {
        for (const auto& element : innerVec) {
            result[currentIndex++] = element;
        }
    }

    data = result;
}

std::map<int, double> evaluate2(
    faiss::Index& index,
    const std::vector<float>& xq,
    const std::vector<faiss::idx_t>& gt,
    int k,
    size_t nq, 
    double& elpsTime) {

    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    auto start_time = std::chrono::steady_clock::now();
    index.search(nq, xq.data(), k, D.data(), I.data());
    auto end_time = std::chrono::steady_clock::now();

    std::map<int, double> recalls;
    auto prob = 1;
    int sum = 0;
    while (prob <= k) 
    {
        for (auto i = 0; i < nq; i++)
        {
            auto start = i * k;
            auto end = i * k + prob;
            sum += count(I.begin() + start, I.begin() + end, gt[i]);
            // sum += count(I.begin(), I.end(), gt[i]);
        }
        recalls[prob] = (double)sum / (double)nq;
        sum = 0;
        prob *= 10;
    }

    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double time_per_query = elapsed_time / nq;
    elpsTime = time_per_query;
    return recalls;
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {    
    
    double t0 = elapsed();

    std::cout << "load data" << std::endl;

    std::vector<float> xq;
    std::vector<float> xb;
    std::vector<float> xt;
    size_t nq;
    size_t d = 0;
    size_t nb, d2;
    size_t nt, k;
    std::vector<faiss::idx_t> gt;
    
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);
        vecs_read<float>("sift1M/sift_query.fvecs", d, nq, xq);
        printf("[%.3f s] Loading database\n", elapsed() - t0);
        vecs_read<float>("sift1M/sift_base.fvecs", d2, nb, xb);

        assert(d == d2 || !"query does not have the same dimension as the train set");

        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        vecs_read<float>("sift1M/sift_learn.fvecs", d, nt, xt);
        
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        size_t nq2;
        std::vector<int> gt_int;
        vecs_read<int>("sift1M/sift_groundtruth.ivecs", k, nq2, gt_int);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        for (int i = 0; i < k * nq; i++) {
            gt.push_back(static_cast<int64_t>(gt_int[i]));
            // std::cout << "GT: " << gt[i] << "\t" << "GT_INT: " << gt_int[i] << "\n";
        }

    }
       
    std::cout << "============ Exact search" << std::endl;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig flat_config;
    flat_config.device = 0;
    faiss::gpu::GpuIndexFlatL2 index(&res, d, flat_config);
    std::cout << "add vectors to index" << std::endl;
    index.add(nq, xb.data());

    std::cout << "warmup" << std::endl;
    std::vector<float> distance(nq * 123, 0);
    std::vector<faiss::idx_t> indices(nq * 123, -1);
    index.search(nq, xq.data(), 123, distance.data(), indices.data());

    for (auto d : distance) {
        assert(d == std::numeric_limits<float>::max() || "distances overflow!!");
    }

    for (auto i : indices) {
        assert(i == -1 || "indices overflow!!");
    }

    std::cout << "benchmark" << std::endl;
    for (int i = 0; i < 11; i++) {
        int k = 1 << i;
        double elpsTime;
        auto recalls = evaluate2(index, xq, gt, k, nq, elpsTime);
        std::cout << "k=" << k << "\t" << elpsTime << "ms\t" << "R@1" << ": " << recalls[1] << "\tR@10" << ": " << recalls[10]  << "\tR@100" << ": " << recalls[100] << "\n";
    }
    std::cout << "\n";
    std::cout << "============ Approximate search" << std::endl;
    faiss::Index* index_ivf = faiss::index_factory(d, "IVF4096,PQ64");
    faiss::gpu::GpuClonerOptions co;
    co.useFloat16 = true;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = 0;
    config.useFloat16LookupTables = true;
    faiss::gpu::GpuIndexIVFPQ index_gpu(&res, static_cast<faiss::IndexIVFPQ*>(index_ivf), config);

    std::cout << "train" << std::endl;
    index_gpu.train(nt, xt.data());
    std::cout << "add vectors to index" << std::endl;
    index_gpu.add(nq, xb.data());

    std::cout << "warmup" << std::endl;
    index_gpu.search(nq, xq.data(), 123, distance.data(), indices.data());

    std::cout << "benchmark" << std::endl;    
    for (int lk = 0; lk < 10; lk++) {
        int k = 1 << lk;
        index_gpu.nprobe = k;
        double elpsTime;
        auto recalls = evaluate2(index_gpu, xq, gt, 100, nq, elpsTime);
        std::cout << "nprobe= " << k << "\t" << elpsTime << "\tms,\t" << "recalls=" << recalls[1] << "\t" << recalls[10]  << "\t" << recalls[100] << "\n";
    }

    delete index_ivf;

    return 0;
}
