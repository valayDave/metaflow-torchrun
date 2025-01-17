# Metaflow torchrun decorator
This repository implements a plugin to run parallel Metaflow tasks as nodes in a torchrun job which can be submitted to AWS Batch or a Kubernetes cluster.

You can install it with:
```
pip install metaflow-torchrun
```

And then you can import it and use in parallel steps:
```
from metaflow import FlowSpec, step, torchrun_parallel

...
class MinGPT(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @kubernetes(image="<YOUR-REGISTRY>/min-gpt:2", cpu=N_CPU, gpu=N_GPU, memory=MEMORY, disk=DISK)
    @torchrun_parallel(
        torchrun_args={"nproc_per_node": N_GPU},
        entrypoint="main.py" # No changes made to original demo script.
    )
    @step
    def torch_multinode(self):
        ...
    ...
```

## Examples

| Directory | torch script description |
| :--- | ---: |
| [Hello](examples/hello/flow.py) | Each process prints their rank and the world size. |  
| [Tensor pass](examples/tensor-pass/flow.py) | Main process passes a tensor to the workers. |  
| [Torch DDP](examples/torch-ddp/flow.py) | A flow that uses a [script from the torchrun tutorials](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html) on multi-node DDP. |  
| [MinGPT](examples/min-gpt/flow.py) | A flow that runs a [torchrun GPT demo](https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html) that simplifies [Karpathy's minGPT](https://github.com/karpathy/minGPT) in a set of parallel Metaflow tasks each contributing their `@resources`. |