// rife implemented with ncnn library

#include "rife_ops.h"


// #include "warp.comp.hex.h"
// #include "warp_pack4.comp.hex.h"
// #include "warp_pack8.comp.hex.h"

using namespace ncnn;

Gt::Gt()
{
    support_vulkan = false;
    support_inplace = false;
    // pipeline_warp = 0;
    // pipeline_warp_pack4 = 0;
    // pipeline_warp_pack8 = 0;
}

/*
int Warp::create_pipeline(const Option& opt)
{
    if (!vkdev)
        return 0;

    std::vector<vk_specialization_type> specializations(0 + 0);

    // pack1
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(warp_comp_data, sizeof(warp_comp_data), opt, spirv);
            }
        }

        pipeline_warp = new Pipeline(vkdev);
        pipeline_warp->set_optimal_local_size_xyz();
        pipeline_warp->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack4
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(warp_pack4_comp_data, sizeof(warp_pack4_comp_data), opt, spirv);
            }
        }

        pipeline_warp_pack4 = new Pipeline(vkdev);
        pipeline_warp_pack4->set_optimal_local_size_xyz();
        pipeline_warp_pack4->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack8
    if (opt.use_shader_pack8)
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(warp_pack8_comp_data, sizeof(warp_pack8_comp_data), opt, spirv);
            }
        }

        pipeline_warp_pack8 = new Pipeline(vkdev);
        pipeline_warp_pack8->set_optimal_local_size_xyz();
        pipeline_warp_pack8->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
}

int Warp::destroy_pipeline(const Option& opt)
{
    delete pipeline_warp;
    pipeline_warp = 0;

    delete pipeline_warp_pack4;
    pipeline_warp_pack4 = 0;

    delete pipeline_warp_pack8;
    pipeline_warp_pack8 = 0;

    return 0;
}
*/


int Gt::load_param(const ParamDict& pd)
{

    return 0;
}
int Gt::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    // W, H, 1, 1 only
    const Mat& input = bottom_blobs[0];
    const Mat& thres = bottom_blobs[1];

    int w = input.w;
    int h = input.h;
    size_t elemsize = input.elemsize;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            if(input.row(i)[j] > thres.row(i)[j])
                top_blob.row(i)[j] = 1.f;
            else
                top_blob.row(i)[j] = 0.f;
        }
    }

    return 0;
}
