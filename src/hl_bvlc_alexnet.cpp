#include <Halide.h>

using namespace Halide;


int main() {
    Var x("x"), y("y"), c("c"), t("t");
    Var xo("xo"), yo("yo");
    Var xi("xi"), yi("yi");
    Var xy("xy");

    //--- Input parameters -------------------------------------
    ImageParam i_data(type_of<float>(), 3, "data");
    Func data = BoundaryConditions::constant_exterior(i_data,0.0f);
    
    ImageParam w_conv1(type_of<float>(), 4, "w_conv1");
    ImageParam b_conv1(type_of<float>(), 1, "b_conv1");
    
    ImageParam w_conv2(type_of<float>(), 4, "w_conv2");
    ImageParam b_conv2(type_of<float>(), 1, "b_conv2");
    
    ImageParam w_conv3(type_of<float>(), 4, "w_conv3");
    ImageParam b_conv3(type_of<float>(), 1, "b_conv3");
    
    ImageParam w_conv4(type_of<float>(), 4, "w_conv4");
    ImageParam b_conv4(type_of<float>(), 1, "b_conv4");
    
    ImageParam w_conv5(type_of<float>(), 4, "w_conv5");
    ImageParam b_conv5(type_of<float>(), 1, "b_conv5");
    
    ImageParam w_fc6(type_of<float>(), 2, "w_fc6");
    ImageParam b_fc6(type_of<float>(), 1, "b_fc6");
    
    ImageParam w_fc7(type_of<float>(), 2, "w_fc7");
    ImageParam b_fc7(type_of<float>(), 1, "b_fc7");
    
    ImageParam w_fc8(type_of<float>(), 2, "w_fc8");
    ImageParam b_fc8(type_of<float>(), 1, "b_fc8");
    

    //--- Processing pipeline ----------------------------------
    
    // conv1
    // blob size (1, 96, 55, 55)
    // param size (96, 3, 11, 11)
    // input_size = [1, 3, 227, 227]
    // groups = 1
    // stride = 4
    // pad = 0
    
    Func conv1("conv1");
    Func s_conv1("s_conv1");
    RDom r_conv1(0,11, 0, 11, 0, 3, "r_conv1");
    s_conv1(x,y,c) = 0.0f;
    
    s_conv1(x,y,c) += w_conv1(r_conv1.x, r_conv1.y, r_conv1.z,c)
            * data(4*x+r_conv1.x-0,4*y+r_conv1.y-0,r_conv1.z),
    
        
    conv1(x,y,c) = s_conv1(x,y,c) + b_conv1(c);

    // input_size = (1, 96, 55, 55)
    Func conv1_relu("conv1_relu");
    conv1_relu(x,y,c) = max(conv1(x,y,c),0.0f);
    

    // input_size = (1, 96, 55, 55)
    // alpha = 0.0001
    // beta = 0.75
    // local_size = 5
    Func norm1("norm1");
    RDom r_norm1(0,5);
    Func c_norm1("c_norm1");
    c_norm1(x,y,c) = select(
        c >= 0 && c < 96,
        conv1_relu(x,y,clamp(c, 0,95)),
        0.0f
    );
    Func scale_norm1("scale_norm1");
    scale_norm1(x,y,c) = 1.0f;
    scale_norm1(x,y,c) += 
        2e-05f *
        c_norm1(x,y,c+r_norm1-2) *
        c_norm1(x,y,c+r_norm1-2);

    norm1(x,y,c) = conv1_relu(x,y,c)*pow(scale_norm1(x,y,c),-0.75f);

    // blob size (1, 96, 27, 27)
    // input_size = (1, 96, 55, 55)
    // stride = 2
    // ksize = 3
    Func pool1("pool1");
    Func s_pool1("s_pool1");
    RDom r_pool1(0,3, 0, 3, "r_pool1");
    s_pool1(x,y,c) = 0.0f;
    s_pool1(x,y,c) = max(s_pool1(x,y,c), 
        norm1(2*x+r_pool1.x,2*y+r_pool1.y,c));
    pool1(x,y,c) = s_pool1(x,y,c);

    // conv2
    // blob size (1, 256, 27, 27)
    // param size (256, 48, 5, 5)
    // input_size = (1, 96, 27, 27)
    // groups = 2
    // stride = 1
    // pad = 2
    
    Func conv2("conv2");
    Func s_conv2("s_conv2");
    RDom r_conv2(0,5, 0, 5, 0, 48, "r_conv2");
    s_conv2(x,y,c) = 0.0f;
    
    s_conv2(x,y,c) += 
        select(
    
            c >= 0 && c < 128,
                    w_conv2(r_conv2.x, r_conv2.y, r_conv2.z,c)
            * pool1(1*x+r_conv2.x-2,1*y+r_conv2.y-2,0+r_conv2.z),
    
            c >= 128 && c < 256,
                    w_conv2(r_conv2.x, r_conv2.y, r_conv2.z,c)
            * pool1(1*x+r_conv2.x-2,1*y+r_conv2.y-2,48+r_conv2.z),
    
        0.0f);
    
        
    conv2(x,y,c) = s_conv2(x,y,c) + b_conv2(c);

    // input_size = (1, 256, 27, 27)
    Func conv2_relu("conv2_relu");
    conv2_relu(x,y,c) = max(conv2(x,y,c),0.0f);
    

    // input_size = (1, 256, 27, 27)
    // alpha = 0.0001
    // beta = 0.75
    // local_size = 5
    Func norm2("norm2");
    RDom r_norm2(0,5);
    Func c_norm2("c_norm2");
    c_norm2(x,y,c) = select(
        c >= 0 && c < 256,
        conv2_relu(x,y,clamp(c, 0,255)),
        0.0f
    );
    Func scale_norm2("scale_norm2");
    scale_norm2(x,y,c) = 1.0f;
    scale_norm2(x,y,c) += 
        2e-05f *
        c_norm2(x,y,c+r_norm2-2) *
        c_norm2(x,y,c+r_norm2-2);

    norm2(x,y,c) = conv2_relu(x,y,c)*pow(scale_norm2(x,y,c),-0.75f);

    // blob size (1, 256, 13, 13)
    // input_size = (1, 256, 27, 27)
    // stride = 2
    // ksize = 3
    Func pool2("pool2");
    Func s_pool2("s_pool2");
    RDom r_pool2(0,3, 0, 3, "r_pool2");
    s_pool2(x,y,c) = 0.0f;
    s_pool2(x,y,c) = max(s_pool2(x,y,c), 
        norm2(2*x+r_pool2.x,2*y+r_pool2.y,c));
    pool2(x,y,c) = s_pool2(x,y,c);

    // conv3
    // blob size (1, 384, 13, 13)
    // param size (384, 256, 3, 3)
    // input_size = (1, 256, 13, 13)
    // groups = 1
    // stride = 1
    // pad = 1
    
    Func conv3("conv3");
    Func s_conv3("s_conv3");
    RDom r_conv3(0,3, 0, 3, 0, 256, "r_conv3");
    s_conv3(x,y,c) = 0.0f;
    
    s_conv3(x,y,c) += w_conv3(r_conv3.x, r_conv3.y, r_conv3.z,c)
            * pool2(1*x+r_conv3.x-1,1*y+r_conv3.y-1,r_conv3.z),
    
        
    conv3(x,y,c) = s_conv3(x,y,c) + b_conv3(c);

    // input_size = (1, 384, 13, 13)
    Func conv3_relu("conv3_relu");
    conv3_relu(x,y,c) = max(conv3(x,y,c),0.0f);
    

    // conv4
    // blob size (1, 384, 13, 13)
    // param size (384, 192, 3, 3)
    // input_size = (1, 384, 13, 13)
    // groups = 2
    // stride = 1
    // pad = 1
    
    Func conv4("conv4");
    Func s_conv4("s_conv4");
    RDom r_conv4(0,3, 0, 3, 0, 192, "r_conv4");
    s_conv4(x,y,c) = 0.0f;
    
    s_conv4(x,y,c) += 
        select(
    
            c >= 0 && c < 192,
                    w_conv4(r_conv4.x, r_conv4.y, r_conv4.z,c)
            * conv3_relu(1*x+r_conv4.x-1,1*y+r_conv4.y-1,0+r_conv4.z),
    
            c >= 192 && c < 384,
                    w_conv4(r_conv4.x, r_conv4.y, r_conv4.z,c)
            * conv3_relu(1*x+r_conv4.x-1,1*y+r_conv4.y-1,192+r_conv4.z),
    
        0.0f);
    
        
    conv4(x,y,c) = s_conv4(x,y,c) + b_conv4(c);

    // input_size = (1, 384, 13, 13)
    Func conv4_relu("conv4_relu");
    conv4_relu(x,y,c) = max(conv4(x,y,c),0.0f);
    

    // conv5
    // blob size (1, 256, 13, 13)
    // param size (256, 192, 3, 3)
    // input_size = (1, 384, 13, 13)
    // groups = 2
    // stride = 1
    // pad = 1
    
    Func conv5("conv5");
    Func s_conv5("s_conv5");
    RDom r_conv5(0,3, 0, 3, 0, 192, "r_conv5");
    s_conv5(x,y,c) = 0.0f;
    
    s_conv5(x,y,c) += 
        select(
    
            c >= 0 && c < 128,
                    w_conv5(r_conv5.x, r_conv5.y, r_conv5.z,c)
            * conv4_relu(1*x+r_conv5.x-1,1*y+r_conv5.y-1,0+r_conv5.z),
    
            c >= 128 && c < 256,
                    w_conv5(r_conv5.x, r_conv5.y, r_conv5.z,c)
            * conv4_relu(1*x+r_conv5.x-1,1*y+r_conv5.y-1,192+r_conv5.z),
    
        0.0f);
    
        
    conv5(x,y,c) = s_conv5(x,y,c) + b_conv5(c);

    // input_size = (1, 256, 13, 13)
    Func conv5_relu("conv5_relu");
    conv5_relu(x,y,c) = max(conv5(x,y,c),0.0f);
    

    // blob size (1, 256, 6, 6)
    // input_size = (1, 256, 13, 13)
    // stride = 2
    // ksize = 3
    Func pool5("pool5");
    Func s_pool5("s_pool5");
    RDom r_pool5(0,3, 0, 3, "r_pool5");
    s_pool5(x,y,c) = 0.0f;
    s_pool5(x,y,c) = max(s_pool5(x,y,c), 
        conv5_relu(2*x+r_pool5.x,2*y+r_pool5.y,c));
    pool5(x,y,c) = s_pool5(x,y,c);

    // blob size (1, 4096)
    // param size (4096, 9216)
    // input_size = (1, 256, 6, 6)
    Func fc6("fc6");
    Func s_fc6("s_fc6");
    RDom r_fc6(0,6, 0, 6, 0, 256, "r_fc6");
    s_fc6(x) = 0.0f;
    Expr cx_fc6 = r_fc6.x + 6*r_fc6.y + 6*6*r_fc6.z;
    s_fc6(x) += w_fc6(cx_fc6,x)
        * pool5(r_fc6.x,r_fc6.y,r_fc6.z);
    fc6(x) = s_fc6(x) + b_fc6(x);
    

    // input_size = (1, 4096)
    Func fc6_relu("fc6_relu");
    fc6_relu(x) = max(fc6(x),0.0f);
    

    // blob size (1, 4096)
    // param size (4096, 4096)
    // input_size = (1, 4096)
    Func fc7("fc7");
    Func s_fc7("s_fc7");
    RDom r_fc7(0,4096);
    s_fc7(x) = 0.0f;
    s_fc7(x) += w_fc7(r_fc7.x,x)
        * fc6_relu(r_fc7.x);
    fc7(x) = s_fc7(x) + b_fc7(x);
    

    // input_size = (1, 4096)
    Func fc7_relu("fc7_relu");
    fc7_relu(x) = max(fc7(x),0.0f);
    

    // blob size (1, 1000)
    // param size (1000, 4096)
    // input_size = (1, 4096)
    Func fc8("fc8");
    Func s_fc8("s_fc8");
    RDom r_fc8(0,4096);
    s_fc8(x) = 0.0f;
    s_fc8(x) += w_fc8(r_fc8.x,x)
        * fc7_relu(r_fc8.x);
    fc8(x) = s_fc8(x) + b_fc8(x);
    

    // blob size (1, 1000)
    // param size None
    // input_size = (1, 1000)
    RDom r_prob(0, 1000, "r_prob");
    Func max_prob("max_prob");
    max_prob(x) = 0.0f;
    max_prob(0) = max(max_prob(0), fc8(r_prob));
    Func centered_prob("centered_prob");
    centered_prob(x) = fc8(x) - max_prob(0);
    Func exp_prob("prob");
    exp_prob(x) = exp(centered_prob(x)) ;
    Func sum_prob("prob");
    sum_prob(x) = 0.0f;
    sum_prob(0) += exp_prob(r_prob);
    Func prob("prob");
    prob(x) = exp_prob(x)/sum_prob(0);


    //--- Schedule ---------------------------------------------
    int parallel_sz = 1;
    int vector_width = 16;

    conv1.compute_root();
    conv1_relu.compute_root();
    norm1.compute_root();
    pool1.compute_root();
    conv2.compute_root();
    conv2_relu.compute_root();
    norm2.compute_root();
    pool2.compute_root();
    conv3.compute_root();
    conv3_relu.compute_root();
    conv4.compute_root();
    conv4_relu.compute_root();
    conv5.compute_root();
    conv5_relu.compute_root();
    pool5.compute_root();
    fc6.compute_root();
    fc6_relu.compute_root();
    fc7.compute_root();
    fc7_relu.compute_root();
    fc8.compute_root();
    max_prob.compute_root();
    sum_prob.compute_root();
    prob.compute_root();

    //--- Output -----------------------------------------------
    prob.compile_to_file("hl_bvlc_alexnet",
        {i_data
    
    ,w_conv1,b_conv1
    ,w_conv2,b_conv2
    ,w_conv3,b_conv3
    ,w_conv4,b_conv4
    ,w_conv5,b_conv5
    ,w_fc6,b_fc6
    ,w_fc7,b_fc7
    ,w_fc8,b_fc8
        }
    );
    prob.compile_to_lowered_stmt("hl_bvlc_alexnet.html",
        {i_data
    
    ,w_conv1,b_conv1
    ,w_conv2,b_conv2
    ,w_conv3,b_conv3
    ,w_conv4,b_conv4
    ,w_conv5,b_conv5
    ,w_fc6,b_fc6
    ,w_fc7,b_fc7
    ,w_fc8,b_fc8
        }, HTML
    );

    return 0;
}
