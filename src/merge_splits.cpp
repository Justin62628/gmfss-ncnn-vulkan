// rife implemented with ncnn library

#include "rife_ops.h"


// #include "warp.comp.hex.h"
// #include "warp_pack4.comp.hex.h"
// #include "warp_pack8.comp.hex.h"

using namespace ncnn;

MergeSplits::MergeSplits()
{
    support_vulkan = false;
    one_blob_only = true;
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

int MergeSplits::load_param(const ParamDict& pd)
{
    upscale_factor = pd.get(0, 2);

    return 0;
}

int MergeSplits::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = d;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty() || channels != upscale_factor * upscale_factor)
        return -100;

//#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < upscale_factor; sh++)
        {
            for (int sw = 0; sw < upscale_factor; sw++)
            {
                int q;
                q = sh * upscale_factor + sw;

                const Mat s = bottom_blob.channel(q);
                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i + sh * h) + sw * w;
                    const float* sptr = s.channel(p).row(i);
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr++;
                    }
                }
            }
        }

    }

    return 0;
}
