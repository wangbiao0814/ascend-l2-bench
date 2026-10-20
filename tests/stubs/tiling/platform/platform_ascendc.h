#pragma once
#include <cstdint>
namespace platform_ascendc {
enum class CoreMemType { UB };
struct PlatformAscendCManager {
    static PlatformAscendCManager *GetInstance() { static PlatformAscendCManager p; return &p; }
    int GetCoreNumAiv() { return 48; }
    void GetCoreMemSize(CoreMemType, uint64_t &bytes) { bytes = 192 * 1024; }
};
}
