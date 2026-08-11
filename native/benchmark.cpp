// Simple throughput probe for the native accumulation hot path.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

extern "C" void ridm_set_threads(int);
extern "C" int ridm_cpu_feature_flags();
extern "C" void ridm_accumulate_contexts_f32(const int64_t*, int64_t, const float*, const float*, int, int,
                                                const float*, float*, float*);

int main(int argc, char** argv) {
    const int64_t tokens = argc > 1 ? std::atoll(argv[1]) : 1'000'000;
    const int vocab = argc > 2 ? std::atoi(argv[2]) : 20'000;
    const int dim = argc > 3 ? std::atoi(argv[3]) : 300;
    const int window = argc > 4 ? std::atoi(argv[4]) : 5;
    if (tokens <= window || vocab <= 0 || dim <= 0 || window <= 0) return 2;
    std::mt19937 rng(42); std::uniform_int_distribution<int> token(0, vocab - 1);
    std::vector<int64_t> ids(tokens);
    for (auto& id : ids) id = token(rng);
    std::vector<float> vectors(static_cast<size_t>(vocab) * dim, 0.01f);
    std::vector<float> idf(vocab, 1.0f), weights(window, 1.0f), matrix(static_cast<size_t>(vocab) * dim), counts(vocab);
    const auto begin = std::chrono::steady_clock::now();
    ridm_accumulate_contexts_f32(ids.data(), tokens, vectors.data(), idf.data(), dim, window, weights.data(), matrix.data(), counts.data());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    const double rate = (tokens - window) / seconds;
    std::cout << std::fixed << std::setprecision(2)
              << "tokens=" << tokens << " tokens_per_second=" << rate << " seconds=" << seconds
              << " estimate_for_1b_hours=" << (1'000'000'000.0 / rate / 3600.0)
              << " cpu_feature_flags=" << ridm_cpu_feature_flags() << "\n";
}
