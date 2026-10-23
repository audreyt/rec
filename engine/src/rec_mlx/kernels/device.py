from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    chip: str
    architecture: str
    workers: int = 36
    mlp_workers: int = 32
    head_workers: int = 32
    prefill_chunk_size: int = 2048
    specialize_mixed_experts: bool = True


def fitting_prefill_chunk(begin, remaining, cap, heads, binding_limit=2**31 - 1):
    """Largest prefill step whose fp32 [heads, step, kv] scores fit one binding.

    Honeykrisp maxStorageBufferRange is 2**31-1. A 32-head step of 2048 attending
    over 8192 keys is exactly 2**31 bytes and the dispatch is refused. Callers
    on this GPU pass a smaller limit: the per-process VA window is fragmented
    by the ~1.8 GiB weight banks, and a 2 GiB score buffer then fails with
    "Failed to allocate BO VMA".
    """
    if begin < 0 or remaining < 1 or cap < 1 or heads < 1 or binding_limit < heads * 4:
        raise ValueError("prefill chunk inputs must be positive")
    budget = binding_limit // (heads * 4)
    if begin + 1 > budget:
        raise ValueError(
            f"one attention token at position {begin} exceeds the storage-buffer limit"
        )
    lo, hi = 1, min(cap, remaining, budget)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid * (begin + mid) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def profile_for_device(info):
    chips = {
        ("applegpu_g16s", "Apple M4 Pro"): "M4 Pro",
        ("applegpu_g17s", "Apple M5 Pro"): "M5 Pro",
        ("applegpu_g17s", "Apple M5 Max"): "M5 Max",
    }
    architecture = info["architecture"]
    identity = (architecture, info["device_name"])
    if identity not in chips:
        raise ValueError(f"Unsupported resident kernel device: {identity}")
    if chips[identity] == "M5 Max":
        return DeviceProfile(
            chip=chips[identity],
            architecture=architecture,
            workers=76,
            mlp_workers=64,
            head_workers=64,
            prefill_chunk_size=4096,
        )
    return DeviceProfile(
        chip=chips[identity],
        architecture=architecture,
        prefill_chunk_size=4096 if chips[identity] == "M5 Pro" else 2048,
        # The specialized M4 resident shader fails scalar/packet parity.
        specialize_mixed_experts=chips[identity] != "M4 Pro",
    )
