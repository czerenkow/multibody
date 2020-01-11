#include "common.h"

std::vector<const char*> to_c_pointers(const std::vector<std::string>& v) {
    std::vector<const char*> result;
    std::transform(v.begin(), v.end(), std::back_inserter(result), [](const std::string& e) -> const char* {
        return e.c_str();
    });
    return result;
}