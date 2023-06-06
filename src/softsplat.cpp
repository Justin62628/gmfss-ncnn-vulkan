// rife implemented with ncnn library

#include "rife_ops.h"


// #include "warp.comp.hex.h"
// #include "warp_pack4.comp.hex.h"
// #include "warp_pack8.comp.hex.h"

using namespace ncnn;

Softsplat::Softsplat()
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


int Softsplat::load_param(const ParamDict& pd)
{

    return 0;
}
int Softsplat::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    const Mat& input = bottom_blobs[0];
    const Mat& flow = bottom_blobs[1];

    int w = input.w;
    int h = input.h;
    int c = input.c;
    size_t elemsize = input.elemsize;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, c-1, elemsize, opt.blob_allocator);
    // top_blob.fill(0);
    ncnn::Mat ss_blob;
    ss_blob.create(w, h, c, elemsize, opt.blob_allocator);
    ss_blob.fill(0);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        Mat m = ss_blob.channel(q);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                float fltOutputX = j + flow.channel(0).row(i)[j];
                float fltOutputY = i + flow.channel(1).row(i)[j];

                int intNorthwestX = (int)(floor(fltOutputX));
                int intNorthwestY = (int)(floor(fltOutputY));
                int intNortheastX = intNorthwestX + 1;
                int intNortheastY = intNorthwestY;
                int intSouthwestX = intNorthwestX;
                int intSouthwestY = intNorthwestY + 1;
                int intSoutheastX = intNorthwestX + 1;
                int intSoutheastY = intNorthwestY + 1;

                float fltNorthwest = ((float)(intSoutheastX)-fltOutputX) * ((float)(intSoutheastY)-fltOutputY);
                float fltNortheast = (fltOutputX - (float)(intSouthwestX)) * ((float)(intSouthwestY)-fltOutputY);
                float fltSouthwest = ((float)(intNortheastX)-fltOutputX) * (fltOutputY - (float)(intNortheastY));
                float fltSoutheast = (fltOutputX - (float)(intNorthwestX)) * (fltOutputY - (float)(intNorthwestY));

                if (intNorthwestX >= 0 && intNorthwestX < w && intNorthwestY >= 0 && intNorthwestY < h)
                    m.row(intNorthwestY)[intNorthwestX] += input.channel(q).row(i)[j] * fltNorthwest;
                if (intNortheastX >= 0 && intNortheastX < w && intNortheastY >= 0 && intNortheastY < h)
                    m.row(intNortheastY)[intNortheastX] += input.channel(q).row(i)[j]*fltNortheast;
                if (intSouthwestX >= 0 && intSouthwestX < w && intSouthwestY >= 0 && intSouthwestY < h)
                    m.row(intSouthwestY)[intSouthwestX] += input.channel(q).row(i)[j] * fltSouthwest;
                if (intSoutheastX >= 0 && intSoutheastX < w && intSoutheastY >= 0 && intSoutheastY < h)
                    m.row(intSoutheastY)[intSoutheastX] += input.channel(q).row(i)[j] * fltSoutheast;
            }
        }
    }
    // top_blob = ss_blob.channel_range(0, c-1) / (ss_blob.channel(-1) + 0.00001);
#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c - 1; q++)
    {
        Mat m = top_blob.channel(q);
        Mat n = ss_blob.channel(q);
        Mat d = ss_blob.channel(c - 1);  // last channel
        for (int i = 0; i < h; i++)
        {
            float* ptr = m.row(i);
            const float* ptr1 = n.row(i);
            const float* ptr2 = d.row(i);
            for (int j = 0; j < w; j++)
            {
                ptr[j] = ptr1[j] / (ptr2[j] + 0.00001);
            }
        }
    }

    return 0;
}
