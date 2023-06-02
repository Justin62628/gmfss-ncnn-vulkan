// rife implemented with ncnn library

#include "rife_ops.h"


// #include "warp.comp.hex.h"
// #include "warp_pack4.comp.hex.h"
// #include "warp_pack8.comp.hex.h"

using namespace ncnn;


ConvexUpsampling::ConvexUpsampling()
{
    support_vulkan = false;
}


int ConvexUpsampling::load_param(const ParamDict& pd)
{
    split_num = pd.get(0, 4);

    return 0;
}

int ConvexUpsampling::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    const Mat& flow = bottom_blobs[0];
    const Mat& up_flow = bottom_blobs[1];

    int w = flow.w;
    int h = flow.h;
    int c = flow.c; // 2
    size_t elemsize = flow.elemsize;

    int outw = w * split_num;
    int outh = h * split_num;
    int outc = c;
    
    Mat& top_blob = top_blobs[0];
    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        /*int p = q / (split_num * split_num);
        int sh = (q % (split_num * split_num)) / split_num;
        int sw = (q % (split_num * split_num)) % split_num;*/

        Mat m = top_blob.channel(q);
        for (int sh = 0; sh < split_num; sh++)
        {
            for (int sw = 0; sw < split_num; sw++)
            {
                const float* sptr = up_flow.channel(q).row(sh * split_num + sw);
                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * split_num + sh) + sw;
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr += split_num;
                    }
                }
            }
        }

    }

    return 0;
}
