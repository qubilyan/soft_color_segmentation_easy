
#include "guidedfilter.h"

static cv::Mat boxfilter(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class GuidedFilterImpl
{
public:
    virtual ~GuidedFilterImpl() {}

    cv::Mat filter(const cv::Mat &p, int depth);

protected:
    int Idepth;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl
{
public:
    GuidedFilterMono(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    int r;
    double eps;
    cv::Mat I, mean_I, var_I;
};

class GuidedFilterColor : public GuidedFilterImpl
{
public:
    GuidedFilterColor(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    std::vector<cv::Mat> Ichannels;
    int r;
    double eps;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat GuidedFilterImpl::filter(const cv::Mat &p, int depth)
{
    cv::Mat p2 = convertTo(p, Idepth);

    cv::Mat result;
    if (p.channels() == 1)
    {
        result = filterSingleChannel(p2);
    }
    else
    {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        for (std::size_t i = 0; i < pc.size(); ++i)
            pc[i] = filterSingleChannel(pc[i]);

        cv::merge(pc, result);
    }

    return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    mean_I = boxfilter(I, r);
    cv::Mat mean_II = boxfilter(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);