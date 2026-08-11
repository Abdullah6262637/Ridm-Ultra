// RIDM hot-path kernels. Public ABI is intentionally Python-header free;
// backend.py binds it via ctypes, making the kernel portable and low-overhead.
#include <algorithm>
#include <cstdint>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_WIN32)
#define RIDM_EXPORT extern "C" __declspec(dllexport)
#else
#define RIDM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
int g_threads = 0;

bool has_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    int info[4]{}; __cpuidex(info, 0, 0); if (info[0] < 7) return false;
    __cpuidex(info, 1, 0); const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || ((_xgetbv(0) & 6) != 6)) return false;
    __cpuidex(info, 7, 0); return (info[1] & (1 << 5)) != 0;
#endif
#else
    return false;
#endif
}

bool has_avx512() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
#else
    return false; // MSVC path stays AVX2-safe until an explicit dispatcher is added.
#endif
#else
    return false;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
void axpy_avx2(float* dst, const float* src, float alpha, int dim) {
    int j = 0; const __m256 a = _mm256_set1_ps(alpha);
    for (; j + 8 <= dim; j += 8) {
        __m256 x = _mm256_loadu_ps(dst + j);
        x = _mm256_fmadd_ps(_mm256_loadu_ps(src + j), a, x);
        _mm256_storeu_ps(dst + j, x);
    }
    for (; j < dim; ++j) dst[j] += alpha * src[j];
}

__attribute__((target("avx2,fma")))
float dot_avx2(const float* a, const float* b, int dim) {
    __m256 sum = _mm256_setzero_ps(); int j = 0;
    for (; j + 8 <= dim; j += 8) sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j), sum);
    alignas(32) float tmp[8]; _mm256_store_ps(tmp, sum);
    float out = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; j < dim; ++j) out += a[j] * b[j];
    return out;
}

__attribute__((target("avx512f,fma")))
float dot_avx512(const float* a, const float* b, int dim) {
    __m512 sum = _mm512_setzero_ps(); int j = 0;
    for (; j + 16 <= dim; j += 16)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), sum);
    alignas(64) float tmp[16]; _mm512_store_ps(tmp, sum);
    float out = 0.0f; for (float value : tmp) out += value;
    for (; j < dim; ++j) out += a[j] * b[j];
    return out;
}
#endif

inline void axpy(float* dst, const float* src, float alpha, int dim) {
#if defined(__GNUC__) || defined(__clang__)
    if (has_avx2()) { axpy_avx2(dst, src, alpha, dim); return; }
#endif
    for (int j = 0; j < dim; ++j) dst[j] += alpha * src[j];
}

inline float dot(const float* a, const float* b, int dim) {
#if defined(__GNUC__) || defined(__clang__)
    if (has_avx512()) return dot_avx512(a, b, dim);
    if (has_avx2()) return dot_avx2(a, b, dim);
#endif
    float out = 0.0f; for (int j = 0; j < dim; ++j) out += a[j] * b[j]; return out;
}
} // namespace

RIDM_EXPORT void ridm_set_threads(int threads) {
    g_threads = std::max(1, threads);
#if defined(_OPENMP)
    omp_set_num_threads(g_threads);
#endif
}

RIDM_EXPORT int ridm_cpu_feature_flags() {
    int flags = has_avx2() ? 1 : 0;
    if (has_avx512()) flags |= 2;
#if defined(_OPENMP)
    flags |= 4;
#endif
    return flags;
}

RIDM_EXPORT void ridm_accumulate_contexts_f32(const int64_t* ids, int64_t n, const float* vecs,
    const float* idf, int dim, int window, const float* distance_weights, float* matrix, float* counts) {
    if (!ids || n <= window || dim <= 0 || window <= 0) return;
#pragma omp parallel for schedule(static) if(n > 256)
    for (int64_t i = window; i < n; ++i) {
        const int64_t target = ids[i];
        if (target < 0) continue;
        float* row = matrix + target * static_cast<int64_t>(dim);
        for (int distance = 1; distance <= window; ++distance) {
            const int64_t context = ids[i - distance];
            if (context < 0) continue;
            const float* src = vecs + context * static_cast<int64_t>(dim);
            const float scale = distance_weights[distance - 1] * idf[context];
            // A target can occur in several worker chunks: protect each update.
            for (int j = 0; j < dim; ++j) {
#pragma omp atomic update
                row[j] += scale * src[j];
            }
        }
#pragma omp atomic update
        counts[target] += 1.0f;
    }
}

RIDM_EXPORT void ridm_matvec_f32(const float* matrix, const float* vector, float* out, int64_t rows, int cols) {
    if (!matrix || !vector || !out || rows <= 0 || cols <= 0) return;
#pragma omp parallel for schedule(static) if(rows > 64)
    for (int64_t row = 0; row < rows; ++row)
        out[row] = dot(matrix + row * static_cast<int64_t>(cols), vector, cols);
}

RIDM_EXPORT int64_t ridm_sample_logits_f32(const float* logits, int64_t v_size, float temperature, int top_k, float random_val) {
    if (!logits || v_size <= 0) return 0;
    temperature = std::max(1e-4f, temperature);
    top_k = std::max(1, std::min(top_k, static_cast<int>(v_size)));

    std::vector<std::pair<float, int64_t>> candidates(v_size);
#pragma omp parallel for schedule(static) if(v_size > 512)
    for (int64_t i = 0; i < v_size; ++i) {
        candidates[i] = {logits[i] / temperature, i};
    }

    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                      [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                          return a.first > b.first;
                      });

    float max_logit = candidates[0].first;
    float sum_exp = 0.0f;
    std::vector<float> probs(top_k);
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp(candidates[i].first - max_logit);
        sum_exp += probs[i];
    }

    float accum = 0.0f;
    float target = random_val * sum_exp;
    for (int i = 0; i < top_k; ++i) {
        accum += probs[i];
        if (accum >= target) {
            return candidates[i].second;
        }
    }
    return candidates[0].second;
}

