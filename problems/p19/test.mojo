from gpu.host import DeviceContext
from random.random import randint
from layout import Layout, LayoutTensor
from op.embedding import embedding_kernel_coalesced
from math import ceildiv


def main():
    with DeviceContext() as ctx:
        alias batch_size = 8
        alias seq_len = 512
        alias vocab_size = 10000
        alias embed_dim = 512

        alias total_elements = batch_size * seq_len * embed_dim

        var indices = ctx.enqueue_create_buffer[DType.int32](batch_size * seq_len)
        var weights = ctx.enqueue_create_buffer[DType.float32](vocab_size * embed_dim).enqueue_fill(0)
        var output = ctx.enqueue_create_buffer[DType.float32](total_elements).enqueue_fill(0)

        with indices.map_to_host() as indices_host:
            randint[DType.int32](indices_host.unsafe_ptr(), batch_size * seq_len, 0, vocab_size)

        alias indices_layout = Layout.row_major(batch_size, seq_len)
        alias weights_layout = Layout.row_major(vocab_size, embed_dim)
        alias output_layout = Layout.row_major(batch_size, seq_len, embed_dim)

        var indices_tensor = LayoutTensor[mut=True, DType.int32, indices_layout](
            indices.unsafe_ptr()
        )
        var weights_tensor = LayoutTensor[mut=True, DType.float32, weights_layout](
            weights.unsafe_ptr()
        )
        var out_tensor = LayoutTensor[mut=True, DType.float32, output_layout](
            output.unsafe_ptr()
        )

        compiled_kernel = ctx.compile_function[
            embedding_kernel_coalesced[
                indices_layout,
                weights_layout,
                output_layout,
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]
        ]()

        alias THREADS_PER_BLOCK = 256
        var blocks = max(1, ceildiv(total_elements, THREADS_PER_BLOCK))

        ctx.enqueue_function(
            compiled_kernel,
            out_tensor,
            indices_tensor,
            weights_tensor,
            grid_dim=(blocks,),
            block_dim=(THREADS_PER_BLOCK,),
        )

        with output.map_to_host() as out_host:
            for i in range(10):
                print("output", i, out_host[i])