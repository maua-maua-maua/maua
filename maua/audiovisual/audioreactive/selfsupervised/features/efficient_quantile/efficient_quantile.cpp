// from https://github.com/pytorch/pytorch/issues/64947#issuecomment-922343374

#include <torch/extension.h>

namespace torch
{

template <class T>
void recursivePartialSorter(T *data, int64_t startOffset, int64_t size, int64_t *quantileIndexes, int64_t qsize)
{
    if (size <= 0)
        return;

    if (qsize == 1)
    {
        std::nth_element(data + startOffset, data + quantileIndexes[0], data + startOffset + size);
    }
    else // Perhaps can be improved for the qsize==2 case.
    {
        // Can the splitting be more optimal.
        int64_t centerQuantile = qsize / 2;
        int64_t pivot = quantileIndexes[centerQuantile];

        recursivePartialSorter(data, startOffset, size, quantileIndexes + centerQuantile, 1);

        int64_t lower_data_size = pivot - startOffset;
        int64_t upper_data_size = size - lower_data_size;
        if (centerQuantile > 0)
            recursivePartialSorter(data, startOffset, lower_data_size, quantileIndexes, centerQuantile);
        if ((qsize - centerQuantile) > 1) // Redundent if equal to 1.
            recursivePartialSorter(data, pivot, upper_data_size, quantileIndexes + centerQuantile,
                                   qsize - centerQuantile);
    }
}

Tensor interp1(Tensor x, Tensor y, Tensor xl, Tensor xu, Tensor xm, int method)
{
    // Note: x, xl, and xu are sorted and if a belongs to xl or xu it belongs to x too.
    int64_t *xp = x.data_ptr<int64_t>();
    int64_t *xlp = xl.data_ptr<int64_t>();
    int64_t *xup = xu.data_ptr<int64_t>();
    Tensor ylower = torch::empty_like(xl, y.options());
    Tensor yupper = torch::empty_like(xl, y.options());

    AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, y.scalar_type(), "efficient_quantiles_interp1", [&] {
        scalar_t *yp = y.data_ptr<scalar_t>();
        scalar_t *ylowerp = ylower.data_ptr<scalar_t>();
        scalar_t *yupperp = yupper.data_ptr<scalar_t>();

        int64_t i = 0;
        int64_t j = 0;
        int64_t n = xl.numel();
        while (i < n)
        {
            if (xlp[i] == xp[j])
                ylowerp[i++] = yp[j];
            else
                j++;
        }
        i = 0;
        j = 0;
        while (i < n)
        {
            if (xup[i] == xp[j])
                yupperp[i++] = yp[j];
            else
                j++;
        }
    });

    Tensor delta;
    if (method == 3)
    {
        delta = ((xu - xl) > 0).to(ScalarType::Double) / 2;
    }
    else // == 4 linear
    {
        delta = xm - xl;
    }

    Tensor result =
        torch::lerp(ylower.to(ScalarType::Double), yupper.to(ScalarType::Double), delta).to(ylower.options());
    return result;
}

Tensor efficient_quantile(const Tensor *x, const Tensor *q, bool ignore_nan, int interpolation_method)
{
    NoNamesGuard guard;

    // The quantiles must be a 1D list of values
    TORCH_CHECK(q->dim() == 1, "The quantiles must be a 1D array of values.");

    // This operation is only supported on CPU
    TORCH_CHECK(x->is_cpu() && q->is_cpu(), "The quantile computation is only supported on cpu.");

    // Check q
    TORCH_CHECK(q->scalar_type() == ScalarType::Float || q->scalar_type() == ScalarType::Double,
                "The quantiles must be of type 'float' or 'double'.");

    if (q->numel() <= 0) // If there are no quantiles. There is nothing to do.
    {
        return Tensor(torch::empty({0}, q->options()));
    }

    // The algorithm use the fact that q is sorted.
    auto sort_result = q->sort();

    // Must be double to avoid round-off issues when converting to index.
    // Can handle sizes up to 2^53 = 9_007_199_254_740_992 >> Anything that ever fits in RAM.
    Tensor qs = std::get<0>(sort_result).to(ScalarType::Double);
    Tensor qs_index = std::get<1>(sort_result);

    TORCH_CHECK(!(qs[0].lt(0).item<bool>() || qs[-1].lt(0).item<bool>()), "The quantiles must be in the range [0, 1].");

    // Size of the result (bytes)
    int64_t result_size = q->numel() * q->dtype().itemsize();

    int64_t size;
    Tensor partialSortTensor = x->clone();
    ;

    // NaN is only an option for FP.
    if (ignore_nan && (x->is_floating_point()))
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x->scalar_type(), "efficient_quantiles_remove_nan", [&] {
            scalar_t *src_ptr = x->data_ptr<scalar_t>();
            scalar_t *dest_ptr = partialSortTensor.data_ptr<scalar_t>();
            size = 0;
            for (int i = 0; i < x->numel(); i++)
            {
                auto val = src_ptr[i];
                if (!std::isnan(val))
                    dest_ptr[size++] = val;
            }
        });
    }
    else
    {
        size = x->numel();
    }

    // Return nan for all quantiles if x is an empty tensor
    if (size <= 0)
    {
        Tensor result = torch::full_like(*q, std::numeric_limits<double>::quiet_NaN()).to(x->options());
        return result;
    }

    Tensor qi;
    switch (interpolation_method)
    {
    case 0: // Lower
        qi = (qs * (size - 1)).to(ScalarType::Long);
        break;
    case 1: // Higher
        qi = (qs * (size - 1)).ceil().to(ScalarType::Long);
        break;
    case 2: // Nearest
        qi = (qs * (size - 1)).round().to(ScalarType::Long);
        break;
    case 3:
    case 4: // mid point / linear interpolation
    {
        auto combined =
            torch::stack({(qs * (size - 1)).to(ScalarType::Long), (qs * (size - 1)).ceil().to(ScalarType::Long)}, 0);
        auto unique = torch::_unique(combined, true, false);
        qi = std::get<0>(unique);
    }
    break;
    default:
        TORCH_CHECK(false, "Unkown quantile interpolation method.");
    }

    // Recursively partially sort the tensor data.
    AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, x->scalar_type(), "efficient_quantile_cpu", [&] {
        recursivePartialSorter(partialSortTensor.data_ptr<scalar_t>(), 0, size, qi.data_ptr<int64_t>(), qi.numel());
    });

    // Extract the quantiles and reorder them according to the order of the input.
    Tensor result;

    switch (interpolation_method)
    {
    case 0:
    case 1:
    case 2:
        result = torch::empty_like(*q, x->options());
        result.index_put_({qs_index}, partialSortTensor.index({qi}));
        break;
    case 3:
    case 4:
    {
        Tensor computed_values = partialSortTensor.index({qi});
        Tensor qm = qs * (size - 1);
        Tensor ql = qm.to(ScalarType::Long);
        Tensor qu = qm.ceil().to(ScalarType::Long);

        Tensor sorted_result = interp1(qi, partialSortTensor.index({qi}), ql, qu, qm, interpolation_method);
        result = torch::empty_like(*q, x->options());
        result.index_put_({qs_index}, sorted_result);
    }
    break;
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_efficient_quantile", &efficient_quantile,
          "Efficient implementation of torch.quantile() without tensor size limitation");
}

} // namespace torch
