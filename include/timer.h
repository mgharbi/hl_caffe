#pragma once

#include <chrono>
namespace {

using namespace std;

inline chrono::steady_clock::time_point get_time(){
    return chrono::steady_clock::now();
}

inline long get_duration(chrono::steady_clock::time_point start,chrono::steady_clock::time_point end) {
    return chrono::duration_cast<chrono::milliseconds>(end-start).count();
}

} // namespace
