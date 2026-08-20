#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>

struct PackedEdge {
  uint64_t key;  // High 32: Row, Low 32: Col
  int32_t  orig; // Original index

  // Default comparison for std::sort is now extremely fast
  bool operator<( const PackedEdge& other ) const { return key < other.key; }
};

class CSRCore {
public:
  // CSR Topology
  std::vector< int > indptr;
  std::vector< int > indices;
  int                nnz  = 0;
  int                nDof = 0;

  // Assembly Mapping
  std::vector< int32_t > gather_sources;
  std::vector< int32_t > assembly_ptr;

  CSRCore( const int* I, const int* J, int64_t n_pairs, int n_dof ) : nDof( n_dof )
  {
    if ( n_pairs == 0 ) {
      // Empty pattern: indptr is all-zeros, indices/gather_sources/assembly_ptr
      // remain empty. update() is guarded by nnz==0 and is a no-op in that case.
      indptr.assign( nDof + 1, 0 );
      return;
    }

    // --- SAFETY & CONFIG ---
    if ( n_pairs > std::numeric_limits< int32_t >::max() ) {
      throw std::overflow_error( "CSRCore: n_pairs exceeds 32-bit limit." );
    }

    // Determine Partitions based on threads
    int num_threads = omp_get_max_threads();

    // We partition by ROWS.
    // 4x partitions per thread is a good heuristic for load balancing.
    const int    num_partitions     = ( num_threads * 4 > nDof ) ? 1 : num_threads * 4;
    const double rows_per_partition = (double)nDof / num_partitions;

    // --- STEP 1: PARALLEL HISTOGRAM ---
    // Count how many edges fall into each partition.
    // We use thread-local counters to avoid atomic contention.

    std::vector< int64_t > partition_counts( num_partitions, 0 );
    // Thread-local counts avoid atomic contention: each thread accumulates
    // into its own private row, then the results are reduced serially.
    std::vector< std::vector< int64_t > > thread_local_counts( num_threads,
                                                               std::vector< int64_t >( num_partitions, 0 ) );

#pragma omp parallel
    {
      int   tid          = omp_get_thread_num();
      auto& local_counts = thread_local_counts[tid];

#pragma omp for schedule( static )
      for ( int64_t k = 0; k < n_pairs; ++k ) {
        int r    = I[k];
        int p_id = (int)( r / rows_per_partition );
        if ( p_id >= num_partitions )
          p_id = num_partitions - 1;
        local_counts[p_id]++;
      }
    }

    // Reduce thread counts to global partition counts & calculate offsets
    // This matrix transposition (thread x part -> part x thread) allows us
    // to calculate exactly where each thread should write its data.
    // Thread-major layout [thread][partition]: each thread's counters reside on
    // their own cache lines, eliminating false sharing during the parallel scatter.
    std::vector< std::vector< int64_t > > write_offsets( num_threads, std::vector< int64_t >( num_partitions ) );
    std::vector< int64_t >                partition_starts( num_partitions + 1, 0 );

    int64_t current_global_offset = 0;
    for ( int p = 0; p < num_partitions; ++p ) {
      partition_starts[p] = current_global_offset;
      for ( int t = 0; t < num_threads; ++t ) {
        write_offsets[t][p] = current_global_offset;
        current_global_offset += thread_local_counts[t][p];
      }
    }
    partition_starts[num_partitions] = current_global_offset; // Should equal n_pairs

    // --- STEP 2: PARALLEL SCATTER (BUCKETING) ---
    // Pack data into a structure that is fast to sort.
    std::vector< PackedEdge > edges( n_pairs );

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

#pragma omp for schedule( static )
      for ( int64_t k = 0; k < n_pairs; ++k ) {
        int r = I[k];
        int c = J[k];

        // Pack (Row, Col) into 64-bit key
        uint64_t key = ( (uint64_t)r << 32 ) | (uint32_t)c;

        int p_id = (int)( r / rows_per_partition );
        if ( p_id >= num_partitions )
          p_id = num_partitions - 1;

        // Determine write position (no atomics needed — thread-major layout
        // means only this thread writes to write_offsets[tid][*]).
        int64_t pos = write_offsets[tid][p_id]++;

        edges[pos] = { key, (int32_t)k };
      }
    }

    // --- STEP 3: PARALLEL SORT & SYMBOLIC COMPRESSION ---
    // Each thread takes ownership of specific partitions, sorts them,
    // and calculates how many non-zeros (nnz) they will produce.

    std::vector< int32_t > partition_nnz( num_partitions, 0 );

    // Temporary indptr logic: Since indptr is cumulative globally,
    // we first fill it relative to the partition start, then fix it later.
    indptr.assign( nDof + 1, 0 );

#pragma omp parallel for schedule( dynamic, 1 )
    for ( int p = 0; p < num_partitions; ++p ) {
      int64_t start = partition_starts[p];
      int64_t end   = partition_starts[p + 1];

      if ( start == end )
        continue;

      // 3a. Sort the partition
      std::sort( edges.begin() + start, edges.begin() + end );

      // 3b. Symbolic Counting (Local CSR generation)
      // We iterate the sorted edges to count unique (r,c) pairs
      // and fill the indptr counts for rows in this partition.
      //
      // INVARIANT: the partition assignment p_id = r / rows_per_partition maps
      // every COO entry with the same row r to exactly one partition. Therefore
      // the indptr[r+1]++ increments below touch disjoint array positions across
      // different partitions — no synchronization is required.

      int32_t local_nnz = 0;
      if ( end > start ) {
        local_nnz = 1; // First entry is always new

        // Determine Row range for this partition
        // We must be careful: indptr is size nDof+1.
        // We extract row from the key.
        int prev_row = static_cast< int32_t >( edges[start].key >> 32 );

        // Mark first row count
        indptr[prev_row + 1]++;

        for ( int64_t k = start + 1; k < end; ++k ) {
          uint64_t curr_key = edges[k].key;
          uint64_t prev_key = edges[k - 1].key;

          int curr_row = static_cast< int32_t >( curr_key >> 32 );

          // If key is different, it's a new matrix entry
          if ( curr_key != prev_key ) {
            local_nnz++;
            indptr[curr_row + 1]++;
          }
        }
      }
      partition_nnz[p] = local_nnz;
    }

    // --- STEP 4: GLOBAL SCAN (OFFSET CALCULATION) ---
    // 4a. Indptr Prefix Sum (This is tricky)
    // Currently indptr[r+1] holds the COUNT of non-zeros in row r.
    // We need to do a standard cumulative sum over the whole array.
    // std::partial_sum is efficient enough here (serial, but cache-linear and fast for <10M rows).
    // For massive arrays, this can be parallelized, but usually not needed.
    std::partial_sum( indptr.begin(), indptr.end(), indptr.begin() );

    // 4b. NNZ Offsets for the other arrays
    std::vector< int32_t > global_nnz_offsets( num_partitions, 0 );
    int32_t                current_nnz = 0;
    for ( int p = 0; p < num_partitions; ++p ) {
      global_nnz_offsets[p] = current_nnz;
      current_nnz += partition_nnz[p];
    }
    this->nnz = current_nnz;

    // --- STEP 5: FINAL FILL ---
    indices.resize( nnz );
    gather_sources.resize( n_pairs );
    assembly_ptr.resize( nnz + 1 );
    assembly_ptr[nnz] = static_cast< int32_t >( n_pairs ); // Sentinel; safe: guarded by INT32_MAX check above

#pragma omp parallel for schedule( dynamic, 1 )
    for ( int p = 0; p < num_partitions; ++p ) {
      int64_t start = partition_starts[p];
      int64_t end   = partition_starts[p + 1];
      if ( start == end )
        continue;

      int32_t write_idx = global_nnz_offsets[p];

      // Fill first entry of the partition
      indices[write_idx]      = static_cast< int32_t >( edges[start].key & 0xFFFFFFFFu );
      assembly_ptr[write_idx] = start;
      gather_sources[start]   = edges[start].orig;

      int32_t internal_count = 0;

      for ( int64_t k = start + 1; k < end; ++k ) {
        uint64_t curr_key = edges[k].key;
        uint64_t prev_key = edges[k - 1].key;

        gather_sources[k] = edges[k].orig;

        if ( curr_key != prev_key ) {
          internal_count++;
          // Close previous assembly pointer
          // (assembly_ptr[i+1] is start of next, which is current k)
          assembly_ptr[write_idx + internal_count] = k;

          // Write new column index
          indices[write_idx + internal_count] = static_cast< int32_t >( curr_key & 0xFFFFFFFFu );
        }
      }
    }
  }

  void update( const double* V_data, double* csr_data ) const
  {
    if ( nnz == 0 )
      return;

#pragma omp      parallel for schedule( static )
    for ( int i = 0; i < nnz; ++i ) {
      int32_t start = assembly_ptr[i];
      int32_t end   = assembly_ptr[i + 1];

      double sum = 0.0;
// The compiler will autovectorize this reduction effectively
// because gather_sources is contiguous in memory now.
#pragma omp simd reduction( + : sum )
      for ( int32_t k = start; k < end; ++k ) {
        sum += V_data[gather_sources[k]];
      }
      csr_data[i] = sum;
    }
  }
};


// ---------------------------------------------------------------------------------------------
// Direct-to-CSR assembly.
//
// CSRCore stages values in a VIJ array of length sizeVIJ and then *gathers* duplicates into CSR.
// That staging array is the dominant memory cost of the meshfree assembly (12.15 GB at 43k DOF,
// alongside I, J and gather_sources), and it exists only because assembly and reduction are
// separated in time.
//
// This assembler removes it by *scattering* instead: each entry knows its CSR slot directly, as
// indptr[row] + offset, where offset is the position of its column within that row. Rows are
// ~1700-2200 long for RKPM stencils (set by the support radius, not by the DOF count), so the
// offset fits in uint16 -- half the width of the int32 gather index it replaces, and it makes the
// value array unnecessary entirely.
//
// Reduction is privatised rather than atomic. Measured on a 708M-op scatter at this problem's
// contention: privatised beats atomics by 2.13x / 1.94x / 1.57x at 4 / 8 / 16 threads, and unlike
// atomics it keeps the summation order fixed, so results stay reproducible. The cost is
// nThreads * nnz * 8 bytes of private buffer, which is still less than the VIJ array it replaces
// at practical thread counts -- but it grows with thread count, so at very high counts the
// trade-off reverses.
struct CSRDirectAssembler {
  const int*            indptr;      // borrowed from the shared pattern
  const int*            indices;     // borrowed
  int                   nnz  = 0;
  int                   nDof = 0;
  int64_t               nPairs = 0;
  int                   nThreads = 1;
  std::vector< uint16_t > offsets;   // within-row offset per entry; the whole map
  std::vector< std::vector< double > > priv;

  // Builds the map against an existing pattern, so there is exactly one definition of what the
  // CSR pattern is and the two assembly paths cannot drift apart.
  CSRDirectAssembler( const int* indptr_,
                      const int* indices_,
                      int        nnz_,
                      int        nDof_,
                      const int* I,
                      const int* J,
                      int64_t    nPairs_,
                      int        nThreads_ )
    : indptr( indptr_ ), indices( indices_ ), nnz( nnz_ ), nDof( nDof_ ), nPairs( nPairs_ ),
      nThreads( nThreads_ > 0 ? nThreads_ : 1 )
  {
    offsets.resize( nPairs );
    int64_t tooWide = 0;

#pragma omp parallel for schedule( static ) reduction( + : tooWide )
    for ( int64_t k = 0; k < nPairs; ++k ) {
      const int  r     = I[k];
      const int  start = indptr[r];
      const int  end   = indptr[r + 1];
      const int* found = std::lower_bound( indices + start, indices + end, J[k] );
      if ( found == indices + end || *found != J[k] ) {
        tooWide += 1;   // reported as a build failure below; no entry may be unmapped
        offsets[k] = 0;
        continue;
      }
      const int64_t off = found - ( indices + start );
      if ( off > std::numeric_limits< uint16_t >::max() ) {
        tooWide += 1;
        offsets[k] = 0;
        continue;
      }
      offsets[k] = static_cast< uint16_t >( off );
    }

    if ( tooWide > 0 )
      throw std::runtime_error( "CSRDirectAssembler: entry unmapped, or within-row offset exceeds "
                                "the 16-bit range (row longer than 65535); a uint32 offset variant "
                                "is required for this pattern." );

    priv.resize( this->nThreads );
    for ( int t = 0; t < this->nThreads; ++t )
      priv[t].assign( nnz, 0.0 );
  }

  // Scatter a whole VIJ value array through the map. This is the equivalence path used to validate
  // the addressing against CSRCore::update; the fused path hands each entity's block straight in.
  void assembleFromVIJ( const double* V, const int* I, double* csr_data )
  {
#pragma omp parallel num_threads( nThreads )
    {
      const int tid = omp_get_thread_num();
      double*   my  = priv[tid].data();
      std::fill( my, my + nnz, 0.0 );

#pragma omp for schedule( static )
      for ( int64_t k = 0; k < nPairs; ++k )
        my[indptr[I[k]] + offsets[k]] += V[k];
    }

    const int nT = nThreads;
#pragma omp parallel for schedule( static ) num_threads( nThreads )
    for ( int i = 0; i < nnz; ++i ) {
      double s = 0.0;
      for ( int t = 0; t < nT; ++t )
        s += priv[t][i];
      csr_data[i] = s;
    }
  }

  // ---- fused path -------------------------------------------------------------------------
  //
  // beginAssembly() / scatterBlock() per entity / reduce(). The entity writes its dense nDof x nDof
  // block into a small thread-local scratch buffer -- which stays in cache -- and scatterBlock
  // pushes it straight into that thread's private CSR copy. No sizeVIJ-sized value array is ever
  // materialized, which is the entire point: at 43k DOF that array alone is 12.15 GB.
  //
  // Registration gives each entity its slice of the map. rowsOfEntity is derived from I: for local
  // row a, every pair (a, b) shares the same global row, so the row is I[mapStart + a * nDofE] and
  // only nDofE values per entity need keeping rather than nDofE^2.
  std::vector< int64_t > entityMapStart;   // into offsets
  std::vector< int64_t > entityDofStart;   // into entityRows
  std::vector< int >     entityNDof;
  std::vector< int >     entityRows;       // global row per local row, concatenated
  std::vector< std::vector< int > > rowStartScratch;   // per thread, size maxNDof

  void registerEntities( const int64_t* mapStarts, const int* nDofs, int nEntities, const int* I )
  {
    entityMapStart.assign( mapStarts, mapStarts + nEntities );
    entityNDof.assign( nDofs, nDofs + nEntities );
    entityDofStart.resize( nEntities );
    int64_t total = 0;
    for ( int e = 0; e < nEntities; ++e ) {
      entityDofStart[e] = total;
      total += nDofs[e];
    }
    entityRows.resize( total );
    // The entity block is stored COLUMN-major: initializeVIJContribution writes
    // I[k] = idcs[k % nDof] and J[k] = idcs[k / nDof], so the row index varies fastest and the
    // first nDof entries of I are exactly the entity's global row list.
#pragma omp parallel for schedule( static )
    for ( int e = 0; e < nEntities; ++e ) {
      const int64_t ms = entityMapStart[e];
      const int     nd = entityNDof[e];
      for ( int a = 0; a < nd; ++a )
        entityRows[entityDofStart[e] + a] = I[ms + a];
    }

    int maxNDof = 0;
    for ( int e = 0; e < nEntities; ++e )
      maxNDof = std::max( maxNDof, nDofs[e] );
    rowStartScratch.assign( nThreads, std::vector< int >( maxNDof, 0 ) );
  }

  void beginAssembly()
  {
#pragma omp parallel num_threads( nThreads )
    {
      double* my = priv[omp_get_thread_num()].data();
      std::fill( my, my + nnz, 0.0 );
    }
  }

  // Called from inside the entity loop, once per entity, by the thread that computed the block.
  // Race-free by construction: each thread owns its private CSR copy.
  void scatterBlock( int tid, int entity, const double* block ) noexcept
  {
    double* __restrict my = priv[tid].data();
    const int64_t ms = entityMapStart[entity];
    const int64_t ds = entityDofStart[entity];
    const int     nd = entityNDof[entity];

    // indptr[row] for each local row, resolved once per entity into a small cache-resident array:
    // the block is column-major, so the row index varies along the inner loop and the lookup cannot
    // be hoisted out of it directly.
    int* __restrict rowStart = rowStartScratch[tid].data();
    for ( int a = 0; a < nd; ++a )
      rowStart[a] = indptr[entityRows[ds + a]];

    // Traverse by ROW even though the block is column-major. The alternative -- following the
    // block's own column-major order -- makes the inner loop write into nDof *different* CSR rows
    // (~375 cache lines, revisited every column), which measured no faster than the gather it is
    // meant to replace. Iterating rows outer confines all writes of an inner loop to a single CSR
    // row (~8-16 kB, cache-resident once touched); the price is a stride-nDof read of the block and
    // of the offsets, both of which are small enough to stay in cache (1.12 MB and 281 kB at
    // nDof = 375).
    // Traverse by ROW, even though the block is stored column-major.
    //
    // A/B measured on a 478M-op assembly (15,120 DOF, block resident, single thread):
    //   row-outer     1562 ms   <- this
    //   column-major  1873 ms   (sequential reads, but writes spread over nDof CSR rows)
    // Row-outer confines the writes of each inner loop to a single CSR row and pays a stride-nDof
    // read of the block and the offsets instead; both are small enough to stay in cache
    // (1.12 MB and 281 kB at nDof = 375), whereas the scattered writes are not.
    //
    // For reference the gather it replaces takes 1866 ms with 16 threads, because it must
    // random-access the 3.83 GB VIJ value array while this random-accesses only the ~164 MB CSR
    // array -- a ~23x smaller working set, which is where the advantage actually comes from.
    const uint16_t* off0 = offsets.data() + ms;
    for ( int a = 0; a < nd; ++a ) {
      double* __restrict row = my + rowStart[a];
      for ( int b = 0; b < nd; ++b ) {
        const int64_t k = static_cast< int64_t >( b ) * nd + a;
        row[off0[k]] += block[k];
      }
    }
  }

  void reduce( double* csr_data )
  {
    const int nT = nThreads;
#pragma omp parallel for schedule( static ) num_threads( nThreads )
    for ( int i = 0; i < nnz; ++i ) {
      double s = 0.0;
      for ( int t = 0; t < nT; ++t )
        s += priv[t][i];
      csr_data[i] = s;
    }
  }

  // Bytes held by the map plus the private buffers -- the figure to compare against the VIJ array.
  int64_t memoryBytes() const
  {
    return static_cast< int64_t >( offsets.size() ) * 2 + static_cast< int64_t >( nThreads ) * nnz * 8
           + static_cast< int64_t >( entityRows.size() ) * 4;
  }
};
