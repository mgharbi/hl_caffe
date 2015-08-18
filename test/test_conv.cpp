#include "gtest/gtest.h"
#include "hl/hl_bvlc_alexnet.h"
#include "halide_image.h"
#include "halide_image_io.h"
#include "timer.h"
#include "log.h"
#include "caffe_io.h"

using namespace Halide::Tools;

class ConvTest : public testing::Test
{
protected:
    ConvTest() {};
    virtual ~ConvTest () {};
};

TEST_F(ConvTest, simpleConvolution){
    Image<float> input;
    load("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/input.png", &input);
    Image<float> output(1000);

    
    Image<float> w_conv1 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_conv1.npy");
    Image<float> b_conv1 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_conv1.npy");
    
    Image<float> w_conv2 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_conv2.npy");
    Image<float> b_conv2 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_conv2.npy");
    
    Image<float> w_conv3 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_conv3.npy");
    Image<float> b_conv3 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_conv3.npy");
    
    Image<float> w_conv4 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_conv4.npy");
    Image<float> b_conv4 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_conv4.npy");
    
    Image<float> w_conv5 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_conv5.npy");
    Image<float> b_conv5 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_conv5.npy");
    
    Image<float> w_fc6 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_fc6.npy");
    Image<float> b_fc6 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_fc6.npy");
    
    Image<float> w_fc7 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_fc7.npy");
    Image<float> b_fc7 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_fc7.npy");
    
    Image<float> w_fc8 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/w_fc8.npy");
    Image<float> b_fc8 = load_net_params("/Users/mgharbi/Documents/projects/hl_caffe/data/bvlc_alexnet/params/b_fc8.npy");
    

    auto start = get_time();
    hl_bvlc_alexnet(
        input
    
    ,w_conv1,b_conv1
    ,w_conv2,b_conv2
    ,w_conv3,b_conv3
    ,w_conv4,b_conv4
    ,w_conv5,b_conv5
    ,w_fc6,b_fc6
    ,w_fc7,b_fc7
    ,w_fc8,b_fc8
    ,output);
    PRINT("- simple convolution:\t %ldms\n", get_duration(start,get_time()));

    save(output, "/Users/mgharbi/Documents/projects/hl_caffe/output/bvlc_alexnet/output.png");
}

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}