from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite


alias SIZE = 249
alias TPB = 128
alias BLOCKS_PER_GRID = (SIZE + TPB - 1) // TPB
alias THREADS_PER_BLOCK = TPB
alias layout_in = Layout.row_major(SIZE)


fn gpu_max_kernel[
    lahout_out: Layout,
    layout_in: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, lahout_out],
    input: LayoutTensor[mut=False, dtype, layout_in],
):
    var global_i = block_idx.x * block_dim.x + thread_idx.x
    var local_i = thread_idx.x

    var shared_max = tb[dtype]().row_major[TPB]().shared().alloc().fill(min_finite[dtype]())

    if global_i < input_size:
        shared_max[local_i] = input[global_i]
    
    barrier()

    var stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared_max[local_i] = max(shared_max[local_i], shared_max[local_i + stride])
        barrier()
        stride = stride // 2

    if local_i == 0:
        output[block_idx.x] = shared_max[0]


fn gpu_sum_kernel[
    lahout_out: Layout,
    layout_in: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, lahout_out],
    input: LayoutTensor[mut=False, dtype, layout_in],
    global_max: Scalar[dtype],
):
    var global_i = block_idx.x * block_dim.x + thread_idx.x
    var local_i = thread_idx.x

    var shared_sum = tb[dtype]().row_major[TPB]().shared().alloc().fill(0)

    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(input[global_i] - global_max))
    shared_sum[local_i] = exp_val
    barrier()

    var stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared_sum[local_i] = shared_sum[local_i] + shared_sum[local_i + stride]
        barrier()
        stride = stride // 2

    if local_i == 0:
        output[block_idx.x] = shared_sum[0]


fn gpu_norm_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
    global_max: Scalar[dtype],
    global_sum: Scalar[dtype],
):
    var global_i = block_idx.x * block_dim.x + thread_idx.x

    if global_i < input_size:
        output[global_i] = exp(input[global_i] - global_max) / global_sum
    


# ANCHOR: softmax_cpu_kernel_solution
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    var max_val: Scalar[dtype] = min_finite[dtype]()
    for i in range(input_size):
        max_val = max(max_val, rebind[Scalar[dtype]](input[i]))

    var sum_exp: Scalar[dtype] = 0.0
    for i in range(input_size):
        var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
        output[i] = exp_val
        sum_exp += exp_val

    for i in range(input_size):
        output[i] = output[i] / sum_exp


# ANCHOR_END: softmax_cpu_kernel_solution

fn run_gpu_softmax[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output_tensor: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input_tensor: LayoutTensor[dtype, layout, MutableAnyOrigin],
    ctx: DeviceContextPtr,
) raises:
    gpu_ctx = ctx.get_device_context()
            
    alias layout_max_sum_out = Layout.row_major(BLOCKS_PER_GRID)
    var max_sum_out_buf = gpu_ctx.create_buffer[dtype](BLOCKS_PER_GRID)
    var max_sum_out = LayoutTensor[dtype, layout_max_sum_out, MutableAnyOrigin](max_sum_out_buf.unsafe_ptr())

    gpu_ctx.call_function[
        gpu_max_kernel[layout_max_sum_out, layout_in, input_size, dtype]
    ](
        max_sum_out,
        input_tensor,
        grid_dim=BLOCKS_PER_GRID,
        block_dim=THREADS_PER_BLOCK,
    )

    gpu_ctx.synchronize()

    var global_max = min_finite[dtype]()

    with max_sum_out_buf.map_to_host() as max_out_host:
        for i in range(len(max_out_host)):
            if max_out_host[i] > global_max:
                global_max = max_out_host[i]


    gpu_ctx.call_function[
        gpu_sum_kernel[layout_max_sum_out, layout_in, input_size, dtype]
    ](
        max_sum_out,
        input_tensor,
        global_max,
        grid_dim=BLOCKS_PER_GRID,
        block_dim=THREADS_PER_BLOCK,
    )

    gpu_ctx.synchronize()

    var global_sum: Scalar[dtype] = 0

    with max_sum_out_buf.map_to_host() as max_out_host:
        for i in range(len(max_out_host)):
            global_sum += max_out_host[i]


    # making sure the output tensor is zeroed out before the kernel is called
    gpu_ctx.memset(
        DeviceBuffer[output_tensor.dtype](
            gpu_ctx,
            rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                output_tensor.ptr
            ),
            input_size,
            owning=False,
        ),
        0,
    )

    gpu_ctx.call_function[
        gpu_norm_kernel[layout_in, input_size, dtype]
    ](
        output_tensor,
        input_tensor,
        global_max,
        global_sum,
        grid_dim=BLOCKS_PER_GRID,
        block_dim=THREADS_PER_BLOCK,
    )

    gpu_ctx.synchronize()


import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout_in, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout_in, MutableAnyOrigin]
        ](input.to_layout_tensor())

        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            run_gpu_softmax[layout_in, input_size, dtype](output_tensor, input_tensor, ctx)

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
