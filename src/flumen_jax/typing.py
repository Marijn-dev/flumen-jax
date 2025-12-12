from jaxtyping import Array, Float, UInt

BatchedOutput = Float[Array, "batch output_dim"]
BatchedState = Float[Array, "batch state_dim"]
BatchedRNNInput = Float[Array, "batch seq_len control_dim+1"]
BatchedTimeIncrement = Float[Array, "batch 1"]
BatchLengths = UInt[Array, "batch 1"]

Inputs = tuple[
    BatchedState, BatchedRNNInput, BatchedTimeIncrement, BatchLengths
]
