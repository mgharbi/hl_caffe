#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include "halide_image.h"
#include "numpy.hpp"

using namespace Halide::Tools;

using namespace std;
using namespace aoba;

Image<float> load_net_params( const string& path )
{

    vector<int> shape(4);
    vector<float> data;
    LoadArrayFromNumpy(path, shape, data);

    int nelts = 1;
    for (int i = 0; i < 4; ++i) {
        if(shape[i] != 0) {
            nelts *= shape[i];
        }
    }

    Image<float> out;
    if(shape[0]*shape[1]*shape[2]*shape[3] > 0) {
        out = Image<float>(shape[3], shape[2], shape[1], shape[0]);
    } else if (shape[0]*shape[1]){
        out = Image<float>(shape[1], shape[0]);
    } else {
        out = Image<float>(shape[0]);
    }
    memcpy(out.data(), data.data(), sizeof(float)*nelts);

    return out;
}
