// rife implemented with ncnn library

#include "rife.h"

#include <algorithm>
#include <vector>
#include "benchmark.h"

#include "rife_preproc.comp.hex.h"
#include "rife_postproc.comp.hex.h"
#include "rife_preproc_tta.comp.hex.h"
#include "rife_postproc_tta.comp.hex.h"
#include "rife_flow_tta_avg.comp.hex.h"
#include "rife_v2_flow_tta_avg.comp.hex.h"
#include "rife_v4_flow_tta_avg.comp.hex.h"
#include "rife_flow_tta_temporal_avg.comp.hex.h"
#include "rife_v2_flow_tta_temporal_avg.comp.hex.h"
#include "rife_v4_flow_tta_temporal_avg.comp.hex.h"
#include "rife_out_tta_temporal_avg.comp.hex.h"
#include "rife_v4_timestep.comp.hex.h"
#include "rife_v4_timestep_tta.comp.hex.h"

#include "rife_ops.h"
#include "layer.h"
#include "layer_type.h"

DEFINE_LAYER_CREATOR(Warp)
DEFINE_LAYER_CREATOR(SplitFeature)
DEFINE_LAYER_CREATOR(MergeSplits)
DEFINE_LAYER_CREATOR(ConvexUpsampling)
DEFINE_LAYER_CREATOR(Softsplat)
DEFINE_LAYER_CREATOR(Gt)


RIFE::RIFE(int gpuid, bool _tta_mode, bool _tta_temporal_mode, bool _uhd_mode, int _num_threads, bool _rife_v2, bool _rife_v4)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    rife_preproc = 0;
    rife_postproc = 0;
    rife_flow_tta_avg = 0;
    rife_flow_tta_temporal_avg = 0;
    rife_out_tta_temporal_avg = 0;
    rife_v4_timestep = 0;
    rife_uhd_downscale_image = 0;
    rife_uhd_upscale_flow = 0;
    rife_uhd_double_flow = 0;
    rife_v2_slice_flow = 0;
    tta_mode = _tta_mode;
    tta_temporal_mode = _tta_temporal_mode;
    uhd_mode = _uhd_mode;
    num_threads = _num_threads;
    rife_v2 = _rife_v2;
    rife_v4 = _rife_v4;
}

RIFE::~RIFE()
{
    // cleanup preprocess and postprocess pipeline
    {
        delete rife_preproc;
        delete rife_postproc;
        delete rife_flow_tta_avg;
        delete rife_flow_tta_temporal_avg;
        delete rife_out_tta_temporal_avg;
        delete rife_v4_timestep;
    }

    if (uhd_mode)
    {
        rife_uhd_downscale_image->destroy_pipeline(flownet.opt);
        delete rife_uhd_downscale_image;

        rife_uhd_upscale_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_upscale_flow;

        rife_uhd_double_flow->destroy_pipeline(flownet.opt);
        delete rife_uhd_double_flow;
    }

    if (rife_v2)
    {
        rife_v2_slice_flow->destroy_pipeline(flownet.opt);
        delete rife_v2_slice_flow;
    }
}

#if _WIN32
static void load_param_model(ncnn::Net& net, const std::wstring& modeldir, const wchar_t* name)
{
    wchar_t parampath[256];
    wchar_t modelpath[256];
    swprintf(parampath, 256, L"%s/%s.param", modeldir.c_str(), name);
    swprintf(modelpath, 256, L"%s/%s.bin", modeldir.c_str(), name);

    {
        FILE* fp = _wfopen(parampath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
        }

        net.load_param(fp);

        fclose(fp);
    }
    {
        FILE* fp = _wfopen(modelpath, L"rb");
        if (!fp)
        {
            fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
        }

        net.load_model(fp);

        fclose(fp);
    }
}
#else
static void load_param_model(ncnn::Net& net, const std::string& modeldir, const char* name)
{
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/%s.param", modeldir.c_str(), name);
    sprintf(modelpath, "%s/%s.bin", modeldir.c_str(), name);

    net.load_param(parampath);
    net.load_model(modelpath);
}
#endif

#if _WIN32
int RIFE::load(const std::wstring& modeldir)
#else
int RIFE::load(const std::string& modeldir)
#endif
{
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = vkdev ? true : false;
    opt.use_fp16_packed = vkdev ? true : false;
    opt.use_fp16_storage = vkdev ? true : false;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = false;

    flownet.opt = opt;
    contextnet.opt = opt;
    fusionnet.opt = opt;

    flownet.set_vulkan_device(vkdev);
    contextnet.set_vulkan_device(vkdev);
    fusionnet.set_vulkan_device(vkdev);

    flownet.register_custom_layer("rife.Warp", Warp_layer_creator);
    flownet.register_custom_layer("model.gmflow.utils.split_feature", SplitFeature_layer_creator);
    flownet.register_custom_layer("model.gmflow.utils.merge_splits", MergeSplits_layer_creator);
    flownet.register_custom_layer("model.gmflow.utils.convex_upsampling", ConvexUpsampling_layer_creator);
    flownet.register_custom_layer("softsplat.forward", Softsplat_layer_creator);
    flownet.register_custom_layer("torch.gt", Gt_layer_creator);
    contextnet.register_custom_layer("rife.Warp", Warp_layer_creator);
    fusionnet.register_custom_layer("rife.Warp", Warp_layer_creator);
    fusionnet.register_custom_layer("softsplat.forward", Softsplat_layer_creator);

#if _WIN32
    load_param_model(flownet, modeldir, L"reuse_576");
    load_param_model(fusionnet, modeldir, L"infer_576");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, L"contextnet");
        load_param_model(fusionnet, modeldir, L"fusionnet");
    }
#else
    load_param_model(flownet, modeldir, "reuse_576");
    load_param_model(fusionnet, modeldir, "infer_576");
    if (!rife_v4)
    {
        load_param_model(contextnet, modeldir, "contextnet");
        load_param_model(fusionnet, modeldir, "fusionnet");
    }
#endif

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_preproc_tta_comp_data, sizeof(rife_preproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_preproc_comp_data, sizeof(rife_preproc_comp_data), opt, spirv);
                }
            }

            rife_preproc = new ncnn::Pipeline(vkdev);
            rife_preproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_postproc_tta_comp_data, sizeof(rife_postproc_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_postproc_comp_data, sizeof(rife_postproc_comp_data), opt, spirv);
                }
            }

            rife_postproc = new ncnn::Pipeline(vkdev);
            rife_postproc->set_optimal_local_size_xyz(8, 8, 3);
            rife_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    if (rife_v4)
    {
        if (vkdev)
        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(rife_v4_timestep_tta_comp_data, sizeof(rife_v4_timestep_tta_comp_data), opt, spirv);
                    else
                        compile_spirv_module(rife_v4_timestep_comp_data, sizeof(rife_v4_timestep_comp_data), opt, spirv);
                }
            }

            std::vector<ncnn::vk_specialization_type> specializations;

            rife_v4_timestep = new ncnn::Pipeline(vkdev);
            rife_v4_timestep->set_optimal_local_size_xyz(8, 8, 1);
            rife_v4_timestep->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int RIFE::process(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (!vkdev)
    {
        // cpu only
        if (rife_v4)
            return process_v4_cpu(in0image, in1image, timestep, outimage);
        /*else
            return process_cpu(in0image, in1image, timestep, outimage);*/
    }

    if (rife_v4)
        return process_v4(in0image, in1image, timestep, outimage);
    return 0;
}

int RIFE::process_v4(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{

    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = flownet.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    ncnn::Mat in0;
    ncnn::Mat in1;
    ncnn::Mat timestep_cpu;
    timestep_cpu.create(1, 1, 1);
    timestep_cpu.fill(timestep);

    if (opt.use_fp16_storage && opt.use_int8_storage)
    {
        in0 = ncnn::Mat(w, h, (unsigned char*)pixel0data, (size_t)channels, 1);
        in1 = ncnn::Mat(w, h, (unsigned char*)pixel1data, (size_t)channels, 1);
    }
    else
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }

    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat in0_gpu;
    ncnn::VkMat in1_gpu;
    ncnn::VkMat timestep_gpu_padded;

    {
        cmd.record_clone(in0, in0_gpu, opt);
        cmd.record_clone(in1, in1_gpu, opt);
        cmd.record_clone(timestep_cpu, timestep_gpu_padded, opt);
    }

    // debug graph variants
    ncnn::Mat debug_54, debug_20, debug_63;
    ncnn::VkMat debug_54_gpu, debug_20_gpu, debug_63_gpu;
    ncnn::Mat in0_cpu, flow01_cpu, flow10_cpu, metric_cpu, feat11_cpu, feat12_cpu, feat13_cpu, feat21_cpu, feat22_cpu, feat23_cpu;    

    ncnn::VkMat out_gpu;

    {
        // preproc
        ncnn::VkMat in0_gpu_padded;
        ncnn::VkMat in1_gpu_padded;
        {
            in0_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in0_gpu;
            bindings[1] = in0_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in0_gpu.w;
            constants[1].i = in0_gpu.h;
            constants[2].i = in0_gpu.cstep;
            constants[3].i = in0_gpu_padded.w;
            constants[4].i = in0_gpu_padded.h;
            constants[5].i = in0_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in0_gpu_padded);
        }
        {
            in1_gpu_padded.create(w_padded, h_padded, 3, in_out_tile_elemsize, 1, blob_vkallocator);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = in1_gpu;
            bindings[1] = in1_gpu_padded;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = in1_gpu.w;
            constants[1].i = in1_gpu.h;
            constants[2].i = in1_gpu.cstep;
            constants[3].i = in1_gpu_padded.w;
            constants[4].i = in1_gpu_padded.h;
            constants[5].i = in1_gpu_padded.cstep;

            cmd.record_pipeline(rife_preproc, bindings, constants, in1_gpu_padded);
        }

        ncnn::VkMat flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23;
        ncnn::VkMat out_gpu_padded;

        {
            // flownet
            ncnn::Extractor ex = flownet.create_extractor();
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("in0", in0_gpu_padded);
            ex.input("in1", in1_gpu_padded);
            ex.extract("out0", flow01, cmd);
            ex.extract("out1", flow10, cmd);
            ex.extract("out2", metric, cmd);
            ex.extract("out3", feat11, cmd);
            ex.extract("out4", feat12, cmd);
            ex.extract("out5", feat13, cmd);
            ex.extract("out6", feat21, cmd);
            ex.extract("out7", feat22, cmd);
            ex.extract("out8", feat23, cmd);
            
            // print_mat_shape(flow01, "flow01");

        }

        // print_mat(flow01, "flow01");

        {
            // infernet
            // load_mat(flow01, "flow01.txt");

            // flow01.create(480, 288, 2);
            // flow01.fill(0.5f);

            ncnn::Extractor ex = fusionnet.create_extractor();
            
            ex.set_blob_vkallocator(blob_vkallocator);
            ex.set_workspace_vkallocator(blob_vkallocator);
            ex.set_staging_vkallocator(staging_vkallocator);

            ex.input("img0", in0_gpu_padded);
            ex.input("img1", in1_gpu_padded);
            ex.input("timestep", timestep_gpu_padded);
            ex.input("flow01", flow01);
            ex.input("flow10", flow10);
            ex.input("metric", metric);
            ex.input("feat11", feat11);
            ex.input("feat12", feat12);
            ex.input("feat13", feat13);
            ex.input("feat21", feat21);
            ex.input("feat22", feat22);
            ex.input("feat23", feat23);
            // debug extract output
            // ex.extract("54", debug_54_gpu, cmd);
            // ex.extract("20", debug_20_gpu, cmd);
            // ex.extract("63", debug_63_gpu, cmd);
            // cmd.record_clone(debug_54_gpu, debug_54, opt);
            // cmd.record_clone(debug_20_gpu, debug_20, opt);
            // cmd.record_clone(debug_63_gpu, debug_63, opt);

            // cmd.record_clone(in0_gpu_padded, in0_cpu, opt);
            // cmd.record_clone(timestep_gpu_padded, timestep_cpu, opt);
            // cmd.record_clone(flow01, flow01_cpu, opt);
            // cmd.record_clone(flow10, flow10_cpu, opt);
            // cmd.record_clone(metric, metric_cpu, opt);
            // cmd.record_clone(feat11, feat11_cpu, opt);
            // cmd.record_clone(feat12, feat12_cpu, opt);
            // cmd.record_clone(feat13, feat13_cpu, opt);
            // cmd.record_clone(feat21, feat21_cpu, opt);
            // cmd.record_clone(feat22, feat22_cpu, opt);
            // cmd.record_clone(feat23, feat23_cpu, opt);

            ex.extract("out0", out_gpu_padded, cmd);
        }

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out_gpu.create(w, h, (size_t)channels, 1, blob_vkallocator);
        }
        else
        {
            out_gpu.create(w, h, channels, (size_t)4u, 1, blob_vkallocator);
        }

        // postproc
        {
            std::vector<ncnn::VkMat> bindings(2);
            bindings[0] = out_gpu_padded;
            bindings[1] = out_gpu;

            std::vector<ncnn::vk_constant_type> constants(6);
            constants[0].i = out_gpu_padded.w;
            constants[1].i = out_gpu_padded.h;
            constants[2].i = out_gpu_padded.cstep;
            constants[3].i = out_gpu.w;
            constants[4].i = out_gpu.h;
            constants[5].i = out_gpu.cstep;

            cmd.record_pipeline(rife_postproc, bindings, constants, out_gpu);
        }
    }

    // download
    {
        ncnn::Mat out;

        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
            out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned char*)outimage.data, (size_t)channels, 1);
        }

        cmd.record_clone(out_gpu, out, opt);

        cmd.submit_and_wait();

        if (!(opt.use_fp16_storage && opt.use_int8_storage))
        {
#if _WIN32
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
            out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
        }
    }

    // debug print output
    // print_mat(timestep_cpu, "timestep", opt);
    // print_mat(debug_54, "debug_54", opt);
    // print_mat(debug_20, "debug_20", opt);
    // print_mat(debug_63, "debug_63", opt);
    // print_mat(in0_cpu, "img0", opt);
    // print_mat(flow01_cpu, "flow01", opt);
    // print_mat(flow10_cpu, "flow10", opt);
    // print_mat(metric_cpu, "metric", opt);
    // print_mat(feat11_cpu, "feat11", opt);
    // print_mat(feat12_cpu, "feat12", opt);
    // print_mat(feat13_cpu, "feat13", opt);
    // print_mat(feat21_cpu, "feat21", opt);
    // print_mat(feat22_cpu, "feat22", opt);
    // print_mat(feat23_cpu, "feat23", opt);


    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}


int RIFE::process_v4_cpu(const ncnn::Mat& in0image, const ncnn::Mat& in1image, float timestep, ncnn::Mat& outimage) const
{
    if (timestep == 0.f)
    {
        outimage = in0image;
        return 0;
    }

    if (timestep == 1.f)
    {
        outimage = in1image;
        return 0;
    }

    const unsigned char* pixel0data = (const unsigned char*)in0image.data;
    const unsigned char* pixel1data = (const unsigned char*)in1image.data;
    const int w = in0image.w;
    const int h = in0image.h;
    const int channels = 3;//in0image.elempack;

//     fprintf(stderr, "%d x %d\n", w, h);

    ncnn::Option opt = flownet.opt;

    // pad to 32n
    int w_padded = (w + 31) / 32 * 32;
    int h_padded = (h + 31) / 32 * 32;

    ncnn::Mat in0;
    ncnn::Mat in1;
    {
#if _WIN32
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
#else
        in0 = ncnn::Mat::from_pixels(pixel0data, ncnn::Mat::PIXEL_RGB, w, h);
        in1 = ncnn::Mat::from_pixels(pixel1data, ncnn::Mat::PIXEL_RGB, w, h);
#endif
    }


    ncnn::Mat out, out_padded;

    // debug
    ncnn::Mat debug_54, debug_20, debug_63;

    {
        // preproc and border padding
        ncnn::Mat in0_padded;
        ncnn::Mat in1_padded;
        ncnn::Mat timestep_padded;
        
        {
            in0_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in0_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in0.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            in1_padded.create(w_padded, h_padded, 3);
            for (int q = 0; q < 3; q++)
            {
                float* outptr = in1_padded.channel(q);

                int i = 0;
                for (; i < h; i++)
                {
                    const float* ptr = in1.channel(q).row(i);

                    int j = 0;
                    for (; j < w; j++)
                    {
                        *outptr++ = *ptr++ * (1 / 255.f);
                    }
                    for (; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
                for (; i < h_padded; i++)
                {
                    for (int j = 0; j < w_padded; j++)
                    {
                        *outptr++ = 0.f;
                    }
                }
            }
        }
        {
            timestep_padded.create(1, 1, 1);  // only multiply with 3d tensor
            timestep_padded.fill(timestep);
        }
        ncnn::Mat flow01, flow10, metric, feat11, feat12, feat13, feat21, feat22, feat23;
        ncnn::Mat out_padded_reversed;

        // load_mat(in0_padded, "in0_padded.txt");
        // load_mat(in1_padded, "in1_padded.txt");


        /* Some Test
        // ncnn::Mat testMat, testOutMat;
        // testMat.create(6, 4, 2);
        // //testOutMat.create(3, 2, 2, 4);
        // int test_cnt = 0;
        // float* testMat_ptr = (float *)testMat.data;
        // for(int c=0;c<2;c++)
        // for(int i = 0; i < 6; i++)
        // {
        //     for(int j = 0; j < 4; j++)
        //     {
        //         *(testMat_ptr++) = test_cnt * 1.f;
        //         test_cnt++;
        //     }
        // }
        
        // ncnn::Layer* op = SplitFeature_layer_creator(nullptr);

        // // set param
        // ncnn::ParamDict pd;
        // pd.set(0, 2);// op_type

        // op->load_param(pd);

        // op->create_pipeline(opt);
        // op->forward(testMat, testOutMat, opt);
        // op->destroy_pipeline(opt);
        // delete op;
        print_mat(testOutMat, "test_out");
        */ 

        {
            // flownet
            ncnn::Extractor ex = flownet.create_extractor();

            ex.input("in0", in0_padded);
            ex.input("in1", in1_padded);
            ex.extract("out0", flow01);
            ex.extract("out1", flow10);
            ex.extract("out2", metric);
            ex.extract("out3", feat11);
            ex.extract("out4", feat12);
            ex.extract("out5", feat13);
            ex.extract("out6", feat21);
            ex.extract("out7", feat22);
            ex.extract("out8", feat23);
            
            // print_mat_shape(flow01, "flow01");

        }

        out_padded.create(w_padded, h_padded, 3);
        // print_mat(flow01, "flow01");
        // print_mat(in0_padded, "img0_cpu", opt);
        // print_mat(timestep_padded, "time_cpu", opt);
        // print_mat(flow01, "flow01_cpu", opt);
        // print_mat(flow10, "flow10_cpu", opt);
        // print_mat(metric, "metric_cpu", opt);
        // print_mat(feat11, "feat11_cpu", opt);
        // print_mat(feat12, "feat12_cpu", opt);
        // print_mat(feat13, "feat13_cpu", opt);
        // print_mat(feat21, "feat21_cpu", opt);
        // print_mat(feat22, "feat22_cpu", opt);
        // print_mat(feat23, "feat23_cpu", opt);

        {
            // infernet
            // load_mat(flow01, "flow01.txt");

            // flow01.create(480, 288, 2);
            // flow01.fill(0.5f);

            ncnn::Extractor ex = fusionnet.create_extractor();
            ex.input("img0", in0_padded);
            ex.input("img1", in1_padded);
            ex.input("timestep", timestep_padded);
            ex.input("flow01", flow01);
            ex.input("flow10", flow10);
            ex.input("metric", metric);
            ex.input("feat11", feat11);
            ex.input("feat12", feat12);
            ex.input("feat13", feat13);
            ex.input("feat21", feat21);
            ex.input("feat22", feat22);
            ex.input("feat23", feat23);
            ex.extract("54", debug_54);
            ex.extract("20", debug_20);
            ex.extract("63", debug_63);
            ex.extract("out0", out_padded);

            // debug print output
            // print_mat(debug_54, "debug_54_cpu", opt);
            // print_mat(debug_20, "debug_20_cpu", opt);
            // print_mat(debug_63, "debug_63_cpu", opt);
            // print_mat(out_padded, "out_cpu", opt);
        }
        // print_mat(out_padded, "out");

        // cut padding and postproc
        out.create(w, h, 3);
        {
            for (int q = 0; q < 3; q++)
            {
                float* outptr = out.channel(q);
                const float* ptr = out_padded.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = *ptr++ * 255.f + 0.5f;
                    }
                }
            }
        }

    }

    // download
    {
#if _WIN32
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB2BGR);
#else
        out.to_pixels((unsigned char*)outimage.data, ncnn::Mat::PIXEL_RGB);
#endif
    }

    // save to file

    return 0;
}
