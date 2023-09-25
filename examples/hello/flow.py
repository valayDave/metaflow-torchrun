from metaflow import FlowSpec, step, torchrun_parallel, current, batch, kubernetes

N_NODES = 2

class HelloTorchrun(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @batch(memory=12228, image="pytorch/pytorch:latest")
    @torchrun_parallel(entrypoint="hi-torchrun.py")
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
    HelloTorchrun()