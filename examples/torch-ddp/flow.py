from metaflow import FlowSpec, step, torchrun_parallel, current, batch, kubernetes, environment

N_NODES = 2
N_GPU = 2

class TorchrunDDP(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @environment(vars = {"NCCL_SOCKET_IFNAME": "eth0"}) 
    @kubernetes(image="eddieob/hello-torchrun:12", gpu=N_GPU)
    @torchrun_parallel(
        entrypoint="multinode_trainer.py", # No changes made to original demo script.
        entrypoint_args={
            "total-epochs": 2,
            "batch-size": 32,
            "save-every": 1
        }
    )
    @step
    def torch_multinode(self):
        print("Optional post-processing from the %s step function." % current.step_name)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass
        
if __name__ == "__main__":
    TorchrunDDP()