#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <limits>

// Include OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// Entry struct (16 Bytes)
struct Entry {
    int row;
    int col;
    int64_t origIdx;

    bool operator<(const Entry& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

class CSRCore {
public:
    std::vector<int> indptr;
    std::vector<int> indices;
    int nnz = 0;
    int nDof = 0;

    std::vector<int64_t> gather_sources;
    std::vector<int64_t> assembly_ptr;

    CSRCore(const int* I, const int* J, int64_t n_pairs, int n_dof) : nDof(n_dof) {

        if (n_pairs == 0) {
            indptr.assign(nDof + 1, 0);
            return;
        }

        // --- CONFIGURATION: Cache-Aware Partitioning ---
        // We want each partition to fit in CPU L2 Cache (approx 256KB - 512KB).
        // Entry is 16 bytes. ~32k entries = 512KB.
        // Formula: partitions = n_pairs / 32000

        int64_t target_partitions = n_pairs / 32000;
        if (target_partitions < 2048) target_partitions = 2048;
        if (target_partitions > 262144) target_partitions = 262144; // Cap at 256k partitions

        const int NUM_PARTITIONS = (int)target_partitions;
        double chunk_size = (nDof > NUM_PARTITIONS) ? (double)nDof / NUM_PARTITIONS : 1.0;

        // --- STEP 1: HISTOGRAM (Serial) ---
        // Fast linear scan. Parallelizing adds atomic overhead, usually not worth it.
        std::vector<int64_t> p_counts(NUM_PARTITIONS, 0);

        for (int64_t k = 0; k < n_pairs; ++k) {
            if (I[k] < 0 || I[k] >= nDof || J[k] < 0 || J[k] >= nDof) {
                throw std::out_of_range("CSRCore Error: Index out of bounds.");
            }
            int p_id = (int)(I[k] / chunk_size);
            if (p_id >= NUM_PARTITIONS) p_id = NUM_PARTITIONS - 1;
            else if (p_id < 0) p_id = 0;
            p_counts[p_id]++;
        }

        // --- STEP 2: OFFSETS ---
        std::vector<int64_t> heads(NUM_PARTITIONS, 0);
        int64_t current_offset = 0;
        for (int i = 0; i < NUM_PARTITIONS; ++i) {
            heads[i] = current_offset;
            current_offset += p_counts[i];
        }
        std::vector<int64_t> p_offsets = heads;
        p_offsets.push_back(current_offset);

        // --- STEP 3: SCATTER (Serial) ---
        std::vector<Entry> flat_buffer(n_pairs);
        for (int64_t k = 0; k < n_pairs; ++k) {
            int r = I[k];
            int p_id = (int)(r / chunk_size);
            if (p_id >= NUM_PARTITIONS) p_id = NUM_PARTITIONS - 1;
            else if (p_id < 0) p_id = 0;

            int64_t pos = heads[p_id]++;
            flat_buffer[pos] = {r, J[k], k};
        }

        // --- STEP 4: PARALLEL SORT ---
        // [PERFORMANCE CRITICAL]
        // Independent partitions allow perfect scaling with OpenMP.
        #pragma omp parallel for schedule(dynamic, 1)
        for (int p = 0; p < NUM_PARTITIONS; ++p) {
            int64_t start = p_offsets[p];
            int64_t end = p_offsets[p+1];
            if (start < end) {
                std::sort(flat_buffer.begin() + start, flat_buffer.begin() + end);
            }
        }

        // --- STEP 5: COMPRESS (Direct Indptr Build) ---
        size_t est_nnz = n_pairs / 2;
        if (est_nnz > 2000000000) est_nnz = 2000000000;

        indices.reserve(est_nnz);
        indptr.resize(nDof + 1); // Use resize to set size immediately

        gather_sources.resize(n_pairs);
        assembly_ptr.reserve(est_nnz + 1);
        assembly_ptr.push_back(0);

        // Process First Entry
        const auto& first = flat_buffer[0];
        indices.push_back(first.col);
        gather_sources[0] = first.origIdx;

        // Initialize Indptr up to the first row found
        // (Handles case where first row is not row 0)
        for (int r = 0; r <= first.row; ++r) indptr[r] = 0;

        int32_t current_nnz = 1; // We just pushed one
        int32_t current_row = first.row;

        for (int64_t k = 1; k < n_pairs; ++k) {
            const auto& curr = flat_buffer[k];
            const auto& prev = flat_buffer[k-1];

            gather_sources[k] = curr.origIdx;

            bool is_new_entry = (curr.row != prev.row) || (curr.col != prev.col);

            if (is_new_entry) {
                assembly_ptr.push_back(k);
                indices.push_back(curr.col);

                // If Row Changed, update indptr
                if (curr.row != current_row) {
                    // Fill indptr for all rows we skipped (usually just current_row + 1)
                    // with the value of current_nnz BEFORE this new entry was added.
                    for (int r = current_row + 1; r <= curr.row; ++r) {
                        indptr[r] = current_nnz;
                    }
                    current_row = curr.row;
                }
                current_nnz++;
            }
        }
        assembly_ptr.push_back(n_pairs);

        // Fill remaining indptr to the end
        for (int r = current_row + 1; r <= nDof; ++r) {
            indptr[r] = current_nnz;
        }

        this->nnz = current_nnz;

        // Safety: Overflow Check
        if (indices.size() > std::numeric_limits<int>::max()) {
             throw std::overflow_error("CSRCore Error: Result Matrix > 2.14B entries.");
        }
    }

    void update(const double* V_data, double* csr_data) const {
        if (nnz == 0) return;

        // --- STEP 6: PARALLEL UPDATE ---
        // [PERFORMANCE CRITICAL]
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nnz; ++i) {
            int64_t start = assembly_ptr[i];
            int64_t end   = assembly_ptr[i+1];

            double sum = 0.0;
            // SIMD Vectorization
            #pragma omp simd reduction(+:sum)
            for (int64_t k = start; k < end; ++k) {
                sum += V_data[gather_sources[k]];
            }
            csr_data[i] = sum;
        }
    }
};
