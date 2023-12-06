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

std::vector<float> mmap_fvecs(const std::string& fname, size_t& d_out, size_t& n_out) {
    std::ifstream file(fname, std::ios::binary); // Open the file in binary mode

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fname << std::endl;
        return {}; // Return an empty vector on error
    }

    // Read all float elements from the file
    std::vector<float> data;
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::cout << "d= " << d << "\n";
    file.seekg(0, std::ios::beg);
    std::vector<float> result;
    std::vector<float> temp(d + 1);
    file.read(reinterpret_cast<char*>(temp.data()),  (d+1)*sizeof(float));
    while (!file.eof()) {
        file.read(reinterpret_cast<char*>(temp.data()),  (d+1)*sizeof(float));
        result.insert(result.end(), temp.begin() + 1, temp.end());
    }
    
    file.close();
    d_out = d;
    n_out = result.size() / d;

    return result;
}

std::vector<int32_t> ivecs_read(const std::string& fname, size_t& d_out, size_t& n_out) {
    std::ifstream file(fname, std::ios::binary); 

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fname << std::endl;
        return {}; 
    }

    std::vector<int32_t> data;
    int32_t temp;
    while (file.read(reinterpret_cast<char*>(&temp), sizeof(int32_t))) {
        data.push_back(temp);
    }

    file.close();
    int32_t d = data[0];
    d_out = d;

    std::vector<int32_t> result;
    for (size_t i = 1; i < data.size(); i += (d + 1)) {
        result.insert(result.end(), data.begin() + i, data.begin() + i + d);
    }
    n_out = result.size() / d;
    
    return result;
}

std::map<int, double> evaluate(
    const faiss::Index& index,
    const std::vector<float>& xq,
    const std::vector<faiss::idx_t>& gt,
    int k,
    size_t nq, 
    double& elpsTime) {

    std::vector<faiss::idx_t> I(nq * k, 0);
    std::vector<float> D(nq * k, 0);

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
            sum += count(I.begin() + start, I.begin() + end, gt[(i + 1) * 100] - 1);
        }
        recalls[prob] = static_cast<double>(sum) / static_cast<double>(nq);
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
        xq = mmap_fvecs("sift1M/sift_query.fvecs", d, nq);
        std::cout << "nq= " << nq << " d= " << d << "\n";

        printf("[%.3f s] Loading database\n", elapsed() - t0);
        xb = mmap_fvecs("sift1M/sift_base.fvecs", d2, nb);
        std::cout << "nb= " << nb << " d2= " << d2 << "\n";

        assert(d == d2 || !"query does not have the same dimension as the train set");

        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        xt = mmap_fvecs("sift1M/sift_learn.fvecs", d, nt);
        std::cout << "nt= " << nt << " d= " << d << "\n";
        
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        size_t nq2;
        std::vector<int32_t> gt_int;
        gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", k, nq2);
        std::cout << "nq2= " << nq2 << " k= " << k << "\n";

        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        for (int i = 0; i < k * nq; i++) {
            gt.push_back(static_cast<int64_t>(gt_int[i]));
        }
        
    }
       
    std::cout << "============ Exact search" << std::endl;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig flat_config;
    flat_config.useFloat16 = true;
    flat_config.device = 0;
    faiss::gpu::GpuIndexFlatL2 index(&res, d, flat_config);
    std::cout << "add vectors to index" << std::endl;
    index.add(nb, xb.data());

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
        auto recalls = evaluate(index, xq, gt, k, nq, elpsTime);
        std::cout << "k=" << k << "\t" << elpsTime << "ms\t" << "R@1" << ": " << recalls[1] << "\n";
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
    index_gpu.add(nb, xb.data());

    std::cout << "warmup" << std::endl;
    index_gpu.search(nq, xq.data(), 123, distance.data(), indices.data());

    std::cout << "benchmark" << std::endl;    
    for (int lk = 0; lk < 10; lk++) {
        int k = 1 << lk;
        index_gpu.nprobe = k;
        double elpsTime;
        auto recalls = evaluate(index_gpu, xq, gt, 100, nq, elpsTime);
        std::cout << "nprobe= " << k << "\t" << elpsTime << "\tms,\t" << "recalls=" << recalls[1] << "\t" << recalls[10]  << "\t" << recalls[100] << "\n";
    }

    delete index_ivf;

    return 0;
}
