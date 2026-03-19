# Project Specification: Mooncake

## Overview

Mooncake is a **KVCache-centric disaggregated LLM inference platform** developed by KVCache.AI.
It enables high-throughput inference by disaggregating the prefill and decoding stages of Large
Language Model serving, and by efficiently managing KV-cache data across multi-tier memory
hierarchies (GPU VRAM, host DRAM, NVMe SSD, CXL devices). On Kimi production workloads it has
demonstrated up to 75% throughput improvement over baseline serving stacks.

---

## 1. Core Technology Stack

### Programming Languages

| Language | Role | Standard |
|----------|------|----------|
| **C++** | Core engine, transports, storage, RPC | C++20 |
| **Python** | Bindings, CLI tools, vLLM connectors, REST APIs | ≥ 3.8 |
| **CUDA / HIP** | GPU memory kernels (EP/PG extensions) | CUDA ≤ 12.9 (primary), 13.x (variant) |
| **Go** | etcd wrapper library | 1.23.8 |
| **CMake** | Build system | ≥ 3.16 |

### Key Libraries & Versions

#### C++ / System

| Library | Version / Notes |
|---------|-----------------|
| **yalantinglibs** | 0.5.7 — async RPC (coro_rpc), coroutine I/O (coro_io), JSON |
| **pybind11** | Bundled in `extern/pybind11` — C++ ↔ Python bindings |
| **ASIO** | Bundled in `mooncake-asio` — async I/O (compiled with `ASIO_SEPARATE_COMPILATION`, `ASIO_DYN_LINK`) |
| **JsonCpp** | Located via `mooncake-common/FindJsonCpp.cmake` |
| **Google Logging (GLOG)** | Located via `mooncake-common/FindGLOG.cmake` |
| **libibverbs / libmlx5** | RDMA/InfiniBand verbs (system package) |
| **NUMA** | libnuma for NUMA-aware memory binding (system package) |
| **Boost** | `boost::functional/hash` used in Store client |
| **etcd-cpp / Go etcd client** | v3.6.1 — optional HA metadata backend |
| **CacheLib allocator** | Custom fork bundled in `mooncake-store/include/cachelib_memory_allocator/` |
| **tl::expected** | Functional error-return type used across Store APIs |

#### Python

| Package | Version / Notes |
|---------|-----------------|
| **aiohttp** | Async HTTP client/server for metadata and REST APIs |
| **requests** | Synchronous HTTP client for metadata operations |
| **setuptools** | ≥ 61.0.0 |
| **wheel** | ≥ 0.37.0 |
| **torch** (optional) | PyTorch — required when building EP/PG CUDA extensions |

#### Development Tools

| Tool | Version |
|------|---------|
| **pre-commit** | 3.7.1 |
| **ruff** | 0.6.9 (Python linting & formatting) |
| **codespell** | 2.2.6 (spell checking) |
| **cmake-format** | 0.6.13 |
| **clang-format** | System package (Google style, 4-space indent) |

---

## 2. Architecture Patterns

### 2.1 Monorepo with Layered Microservice-Like Components

The repository follows a **monorepo** layout. Each `mooncake-*` directory is an independently
buildable CMake sub-project linked via the top-level `CMakeLists.txt`. The components form a
clean layered architecture:

```
┌─────────────────────────────────────────────────────┐
│            Integration Layer                        │
│  mooncake-integration  mooncake-wheel (Python API)  │
├─────────────────────────────────────────────────────┤
│         Compute Parallelism Layer                   │
│      mooncake-ep (EP)    mooncake-pg (PG)           │
├─────────────────────────────────────────────────────┤
│              Storage Layer                          │
│         mooncake-store  mooncake-p2p-store          │
├─────────────────────────────────────────────────────┤
│              Transfer Layer                         │
│         mooncake-transfer-engine (TE / TENT)        │
├─────────────────────────────────────────────────────┤
│               Foundation Layer                      │
│  mooncake-common  mooncake-asio  mooncake-rl        │
└─────────────────────────────────────────────────────┘
```

### 2.2 Component Roles

| Component | Role |
|-----------|------|
| `mooncake-transfer-engine` | Unified, batched, multi-protocol data transfer API (RDMA, TCP, NVLink, CXL, EFA, NVMe-oF, HIP, Ascend, …) |
| `mooncake-store` | Distributed KVCache store — master/replica, eviction, allocation, optional etcd HA |
| `mooncake-p2p-store` | Lightweight peer-to-peer object store (no master) |
| `mooncake-ep` | Expert Parallelism CUDA extension (MoE model support via IBGDA/MLX5 GDA) |
| `mooncake-pg` | Pipeline Gradient parallelism CUDA extension |
| `mooncake-rl` | Reinforcement-learning training support utilities |
| `mooncake-integration` | vLLM (v0.1 and v1), LMCache, SGLang connectors |
| `mooncake-wheel` | Python package (`mooncake-transfer-engine` on PyPI, v0.3.10) — CLI entry points and bindings |
| `mooncake-common` | Shared CMake helpers, JSON/GLOG finders, etcd wrapper |
| `mooncake-asio` | ASIO shared library target |

### 2.3 Transport Plugin Architecture

The Transfer Engine uses a **strategy/plugin pattern**. Each network technology is implemented
as a `Transport` subclass and registered at runtime via `installTransport(proto, args)`.
Supported plugins:

- `tcp` — TCP/IP
- `rdma` — InfiniBand / RoCEv2 / eRDMA / GPUDirect RDMA
- `nvlink` / `intranode_nvlink` — NVLink (inter- and intra-node)
- `hip` — AMD ROCm HIP
- `cxl` — Compute Express Link
- `nvmeof` — NVMe over Fabrics
- `efa` — AWS Elastic Fabric Adapter
- `barex` — Bare DRAM transfers
- `ascend` / `hccl` / `ascend_direct` / `ubshmem` / `heterogeneous_rdma` — Huawei Ascend NPU

### 2.4 Distributed Store: Master / Replica

`mooncake-store` follows a **master–replica** model:

- A single **Master** service (`MasterService`) manages object metadata, placement, allocation
  strategy, eviction policy, and high-availability via optional **etcd** backend.
- **Clients** (`MasterClient`) connect to the master over `coro_rpc` (yalantinglibs RPC) to
  perform Put/Get/Delete/Exist operations.
- Replicas hold actual KVCache data buffers in DRAM or on NVMe.
- A **REST API** (`mc_store_rest_server`) exposes the store to non-C++ callers.

---

## 3. Data Flow

### 3.1 Inference Serving (Disaggregated Prefill / Decode)

```
User Request
     │
     ▼
vLLM / SGLang Scheduler
     │  KVConnectorMetadata
     ▼
MooncakeConnector (mooncake-integration / mooncake-wheel)
     │  TransferRequest batches
     ▼
TransferEngine.submitTransfer()
     │  selects best Transport (RDMA/NVLink/TCP/…)
     ▼
Transport Plugin ──────────────────────────────────┐
     │  DMA / network transfer                     │
     ▼                                             ▼
Remote KVCache Segment                   Local KVCache Segment
  (Prefill node VRAM/DRAM)               (Decode node VRAM/DRAM)
     │
     ▼ (on transfer completion)
MooncakeConnector.finish_async_transfer()
     │
     ▼
Decoding proceeds with transferred KV tensors
```

### 3.2 KVCache Store (Put / Get)

```
Client (Python / C++)
     │  Put(object_key, value, replica_list)
     ▼
MasterClient  ──RPC──►  MasterService
                              │ AllocationStrategy selects segments
                              │ EvictionStrategy frees space if needed
                              ▼
                         ReplicaManager
                              │  TransferEngine.submitTransfer()
                              ▼
                    Segment buffers (DRAM / NVMe / CXL)

     │  Get(object_key)
     ▼
MasterClient  ──RPC──►  MasterService
                              │ Resolves replica locations
                              ▼
                         TransferEngine (pull data to caller)
```

### 3.3 Python Binding Data Path

```
Python caller
     │
     ▼
mooncake Python package (mooncake-wheel)
  └─ _mooncake.so  (pybind11 extension)
       │
       ▼
  C++ TransferEngine / MasterClient
       │
       ▼
  Transport Plugins / Store backend
```

---

## 4. API Contracts / Key Function Signatures

### 4.1 Transfer Engine (C++) — `mooncake-transfer-engine/include/transfer_engine.h`

```cpp
namespace mooncake {

class TransferEngine {
public:
    // Construction
    explicit TransferEngine(bool auto_discover = false);
    TransferEngine(bool auto_discover, const std::vector<std::string>& filter);
    ~TransferEngine();

    // Initialization
    int init(const std::string& metadata_conn_string,
             const std::string& local_server_name,
             const std::string& ip_or_host_name = "",
             uint64_t rpc_port = 12345);

    int freeEngine();

    // Transport management
    Transport* installTransport(const std::string& proto, void** args);
    int        uninstallTransport(const std::string& proto);

    // Segment (memory region) management
    SegmentHandle openSegment(const std::string& segment_name);
    int           closeSegment(SegmentHandle handle);
    int           removeLocalSegment(const std::string& segment_name);
    Status        CheckSegmentStatus(SegmentID sid);

    // Memory registration
    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location = kWildcardLocation,
                            bool remote_accessible = true,
                            bool update_metadata  = true);
    int unregisterLocalMemory(void* addr, bool update_metadata = true);
    int registerLocalMemoryBatch(const std::vector<BufferEntry>& buffer_list,
                                 const std::string& location);
    int unregisterLocalMemoryBatch(const std::vector<void*>& addr_list);

    // Batched transfer operations
    BatchID allocateBatchID(size_t batch_size);
    Status  freeBatchID(BatchID batch_id);
    Status  submitTransfer(BatchID batch_id,
                           const std::vector<TransferRequest>& entries);
    Status  submitTransferWithNotify(
                BatchID batch_id,
                const std::vector<TransferRequest>& entries,
                TransferMetadata::NotifyDesc notify_msg);
    Status  getTransferStatus(BatchID batch_id, size_t task_id,
                              TransferStatus& status);

    // Peer notification
    int getNotifies(std::vector<TransferMetadata::NotifyDesc>& notifies);
    int sendNotifyByID(SegmentID target_id,
                       TransferMetadata::NotifyDesc notify_msg);
    int sendNotifyByName(std::string remote_agent,
                         TransferMetadata::NotifyDesc notify_msg);

    // Network info
    std::string getLocalIpAndPort();
    int         getRpcPort();
};

} // namespace mooncake
```

Key type aliases (all from `Transport`):
```cpp
using TransferRequest    = Transport::TransferRequest;
using TransferStatus     = Transport::TransferStatus;
using TransferStatusEnum = Transport::TransferStatusEnum;
using SegmentHandle      = Transport::SegmentHandle;
using SegmentID          = Transport::SegmentID;
using BatchID            = Transport::BatchID;  // uint64_t; INVALID_BATCH_ID = UINT64_MAX
using BufferEntry        = Transport::BufferEntry;
```

---

### 4.2 Mooncake Store — Master Client (C++) — `mooncake-store/include/master_client.h`

```cpp
namespace mooncake {

class MasterClient {
public:
    explicit MasterClient(const UUID& client_id,
                          MasterClientMetric* metrics = nullptr);
    ~MasterClient();

    // Non-copyable
    MasterClient(const MasterClient&)            = delete;
    MasterClient& operator=(const MasterClient&) = delete;

    // Connection
    [[nodiscard]] ErrorCode Connect(
        const std::string& master_addr = "localhost:50051");

    // Key existence checks
    [[nodiscard]] tl::expected<bool, ErrorCode>
        ExistKey(const std::string& object_key);

    [[nodiscard]] std::vector<tl::expected<bool, ErrorCode>>
        BatchExistKey(const std::vector<std::string>& object_keys);

    // Object lifecycle (Put / Get / Delete — see full header for signatures)
    // All return tl::expected<T, ErrorCode> or ErrorCode
};

} // namespace mooncake
```

Return convention: functions that can fail return `tl::expected<T, ErrorCode>` (success) or
`ErrorCode` directly; callers must check before using the value.

---

### 4.3 Python vLLM v1 Connector — `mooncake-wheel/mooncake/mooncake_connector_v1.py`

```python
class MooncakeConnector(KVConnectorBase_V1):

    # ---- Scheduler-side API ------------------------------------------------
    def get_num_new_matched_tokens(
        self,
        request: "SchedulerRequest",
        num_computed_tokens: int,
    ) -> tuple[int, bool]: ...

    def update_state_after_alloc(
        self,
        request: "SchedulerRequest",
        blocks: "KVConnectorBlocks",
        num_external_tokens: int,
    ) -> None: ...

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "KVConnectorMetadata": ...

    # ---- Worker-side API ---------------------------------------------------
    def prepare_for_forward(
        self,
        num_requests: int,
        scheduler_output: "SchedulerOutput",
    ) -> None: ...

    def finish_async_transfer(self) -> None: ...

    # ---- Lifecycle ---------------------------------------------------------
    def request_finished(self, request_id: str) -> None: ...
    def release_all_seqs(self) -> None: ...
```

---

### 4.4 REST API Entry Points (Python CLI)

| CLI Command | Entry Point | Purpose |
|-------------|-------------|---------|
| `mooncake_master` | `mooncake.cli:main` | Start master RPC service |
| `mooncake_client` | `mooncake.cli_client:main` | CLI client for Put/Get/Delete |
| `transfer_engine_bench` | `mooncake.cli_bench:main` | Transfer bandwidth benchmark |
| `mooncake_http_metadata_server` | `mooncake.http_metadata_server:main` | HTTP metadata service |
| `mc_store_rest_server` | `mooncake.mooncake_store_service:main` | Store REST API server |
| `transfer_engine_topology_dump` | `mooncake.transfer_engine_topology_dump:main` | Network topology introspection |

---

### 4.5 Handshake Protocol

```cpp
enum class HandShakeRequestType : uint8_t {
    Connection  = 0,
    Metadata    = 1,
    Notify      = 2,
    OldProtocol = 0xff,  // backward compatibility
};
```

---

## 5. Coding Style & Constraints

### 5.1 C++ Style (enforced by `.clang-format`)

| Rule | Value |
|------|-------|
| Base style | **Google** |
| Indent width | **4 spaces** (tabs never used) |
| Column limit | **80 characters** |
| `#include` ordering | Not sorted automatically (`SortIncludes: false`) |
| C++ standard | **C++20** |
| Naming — types | `PascalCase` (e.g., `TransferEngine`, `MasterClient`) |
| Naming — functions | `camelCase` (e.g., `submitTransfer`, `openSegment`) |
| Naming — members | `snake_case_` with trailing underscore (e.g., `client_id_`) |
| Naming — constants / macros | `UPPER_SNAKE_CASE` |
| Namespaces | All production code lives in `namespace mooncake { }` |
| Error handling | Return `int` (0 = success, negative = error) **or** `tl::expected<T, ErrorCode>` for richer context; never throw exceptions in hot paths |
| Ownership | RAII via `std::unique_ptr` / `std::shared_ptr`; raw pointers only when non-owning |
| Concurrency | Use coroutines (`coro_rpc`, `async_simple`) for async I/O; `std::mutex` for shared state |
| Logging | Google Logging (`LOG(INFO)`, `LOG(WARNING)`, `LOG(ERROR)`) throughout |
| Pre-conditions | `[[nodiscard]]` on all functions whose return value must be checked |
| Copyright header | Apache 2.0 block required at the top of every source file |

### 5.2 Python Style (enforced by `ruff` v0.6.9)

| Rule | Value |
|------|-------|
| Formatter | **ruff** (replaces black + isort) |
| Line length | 88 characters (ruff default) |
| Naming — variables / functions | `snake_case` |
| Naming — classes | `PascalCase` |
| Type hints | Used in public APIs; encouraged everywhere |
| Async | `asyncio` / `aiohttp` for I/O-bound operations |
| Error handling | Explicit exception handling with specific exception types; no bare `except:` |
| Imports | Grouped (stdlib → third-party → local), sorted by ruff |

### 5.3 CMake Style (enforced by `cmake-format` v0.6.13)

- Option names: `UPPER_SNAKE_CASE` (e.g., `WITH_STORE`, `USE_ETCD`).
- Targets linked with `target_link_libraries(target PRIVATE|PUBLIC dep)`.
- Each `mooncake-*` sub-project has its own `CMakeLists.txt` and can be built standalone.

### 5.4 General Constraints

- **Error handling is mandatory**: every function that can fail must propagate or log the error.
  Silent failures are not acceptable.
- **No C++ exceptions across module boundaries**: the Python C extension and inter-process
  communication layer must not leak C++ exceptions; use return codes instead.
- **Thread safety**: shared data structures accessed by multiple threads must be protected by
  mutexes or lock-free primitives; document thread-safety guarantees in header comments.
- **Memory registration**: all user-space buffers that will be read or written by the Transfer
  Engine must be registered with `registerLocalMemory()` before use, and deregistered
  afterwards. Memory region sizes are clamped to `globalConfig().max_mr_size`.
- **Metadata synchronization**: transports that modify local segment metadata must call
  `synchronizeLocal()` to publish changes to the metadata store.
- **Transport discovery**: when `auto_discover = true`, the engine auto-detects and installs
  applicable transports; an optional filter list restricts which protocols are considered.
- **Build flags**: production builds use `-O3 -g0` (optimized, no debug info); ASAN builds
  add `-fsanitize=address` via `ENABLE_ASAN=ON`.
- **Spell checking**: all source files, comments, and documentation are checked by
  `codespell` v2.2.6 (`.typos.toml` defines project-specific exceptions).
- **Large file guard**: files > the pre-commit threshold are blocked from being committed
  (enforced by `check-added-large-files` hook).

---

## 6. Build & Test Quick Reference

### Build (CMake)

```bash
# Install system dependencies
sudo ./dependencies.sh

# Configure (enable all major components + tests)
cmake -B build \
  -DWITH_TE=ON \
  -DWITH_STORE=ON \
  -DUSE_HTTP=ON \
  -DUSE_ETCD=ON \
  -DSTORE_USE_ETCD=ON \
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_EXAMPLES=ON

cmake --build build -j$(nproc)
```

### Python Package

```bash
pip install -e mooncake-wheel/
```

### Run Unit Tests

```bash
cd build && ctest --output-on-failure
```

### Lint

```bash
pip install -r requirements-dev.txt
pre-commit run --all-files
```

### CI Matrix (`.github/workflows/ci.yml`)

| Dimension | Values |
|-----------|--------|
| Python | 3.10, 3.12 |
| CUDA | 12.8.1 (primary), 13.x (variant) |
| etcd | v3.6.1 |
| OS | Ubuntu (Linux x86-64) |

---

## 7. Repository Structure (Top Level)

```
Mooncake/
├── CMakeLists.txt               # Top-level build (requires CMake ≥ 3.16)
├── dependencies.sh              # One-shot dependency installer
├── requirements-dev.txt         # Python dev tools (pre-commit, ruff, …)
├── requirements_docs.txt        # Sphinx docs dependencies
├── extern/pybind11/             # Bundled pybind11
├── mooncake-asio/               # Bundled ASIO shared library
├── mooncake-common/             # Shared CMake helpers, etcd wrapper
├── mooncake-transfer-engine/    # Transfer Engine (C++) — core
├── mooncake-store/              # Distributed KVCache Store (C++) — v2.0.0
├── mooncake-p2p-store/          # Peer-to-peer Store (C++)
├── mooncake-ep/                 # Expert Parallelism CUDA extension
├── mooncake-pg/                 # Pipeline Gradient CUDA extension
├── mooncake-rl/                 # Reinforcement learning utilities
├── mooncake-integration/        # vLLM / LMCache / SGLang connectors
├── mooncake-wheel/              # Python package v0.3.10
│   ├── mooncake/                # Python modules
│   └── pyproject.toml
├── benchmarks/                  # Performance benchmarks
├── monitoring/                  # Prometheus + Grafana stack
├── docker/                      # Dockerfiles
├── docs/                        # Sphinx documentation source
├── scripts/                     # Utility scripts
└── .github/workflows/           # CI/CD pipelines
```

---

*This document was generated from the repository at commit HEAD on 2026-03-19.
Keep it updated when new components, transports, or APIs are introduced.*
