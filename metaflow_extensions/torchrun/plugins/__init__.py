from metaflow.plugins.parallel_decorator import ParallelDecorator, _local_multinode_control_task_step_func, UBF_CONTROL
from metaflow.exception import MetaflowException
from functools import partial
import subprocess
import socket
import sys
import os


NODE_STARTED_VAR = "torchrun_node_started"


class TorchrunDecoratorParallel(ParallelDecorator):

    name = "torchrun_parallel"
    defaults = {
        "master_port": None,
        "torchrun_args": {
            "nproc_per_node": 1, # could tune this
            "max-restarts": 3,   # could tune this, relate to Metaflow retry_count
        },
        "entrypoint": "trainer.py",
        "entrypoint_args": {},
        "all_nodes_started_timeout": 300
    }
    IS_PARALLEL = True

    def _torchrun_process(self, ubf_context):

        from metaflow import current

        if os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local":
            torchrun_args = {}
        else:

            # collect variables from environment
            if "AWS_BATCH_JOB_ID" in os.environ:
                num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
                node_index = int(os.environ["AWS_BATCH_JOB_NODE_INDEX"])
                if ubf_context == UBF_CONTROL:
                    main_addr = socket.gethostname()
                else:
                    main_ip = os.environ["AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"]
                    main_addr = socket.gethostbyaddr(main_ip)[0]
            else: # kubernetes
                num_nodes = int(os.environ["WORLD_SIZE"])
                node_index = int(os.environ["RANK"])
                main_addr = os.environ["MASTER_ADDR"]

            main_port = "3339"
        
            # create torchrun args
            torchrun_args = {
                "rdzv-id": "123",
                "rdzv_endpoint": "%s:%s" % (main_addr, main_port),
                "nnodes": num_nodes, 
                "master_addr": main_addr,
                "master_port": main_port, # batch doesn't matter. k8s depends on port opened in @kubernetes jobset. 
                "node_rank": node_index,
                "nproc_per_node": 1, # TODO: set to N visible devices on node
                "rdzv-backend": "c10d",
            }

        # apply user overrides in @torchrun_parallel decorator.
        torchrun_args.update(self.attributes['torchrun_args'])

        # setup torchrun part of command
        cmd = ["torchrun"]
        for arg, val in torchrun_args.items():
            cmd.extend(["--%s" % arg, str(val)])

        # setup torch user script part of command
        cmd.append(self.attributes['entrypoint'])
        if isinstance(self.attributes['entrypoint_args'], str):
            cmd.append(self.attributes['entrypoint_args'])
        elif isinstance(self.attributes['entrypoint_args'], dict):
            for arg, val in self.attributes['entrypoint_args'].items():
                cmd.extend(["--%s" % arg, str(val)])
        else:
            raise TypeError("entrypoint_args must be a dict or str")

        self.cmd = " ".join(cmd)
        subprocess.run(cmd, check=True)

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):

        def _torchrun_with_step_func_postprocessing():
            self._ensure_torch_installed()
            self._torchrun_process(ubf_context)
            setattr(flow, "command", " ".join(self.cmd))
            step_func()
        if (
            ubf_context == UBF_CONTROL
            and os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local"
        ):
            from functools import partial
            env_to_use = getattr(self.environment, "base_env", self.environment)
            return partial(
                _local_multinode_control_task_step_func,
                flow,
                env_to_use,
                _torchrun_with_step_func_postprocessing,
                retry_count,
            )
        else:
            return _torchrun_with_step_func_postprocessing

    def _ensure_torch_installed(self):
        try:
            import torch
        except ImportError:
            print("PyTorch is not installed. Installing the latest version of the torch package from PyPi.")
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "torch"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            import torch

def get_backend():
    try:
        import torch
        return torch.distributed.get_backend()
    except ImportError:
        return None

STEP_DECORATORS_DESC = [("torchrun_parallel", ".TorchrunDecoratorParallel")]
