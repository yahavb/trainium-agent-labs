# Environment report

Generated 2026-10-10 18:32:24 UTC on seat-230

## neuron-ls

```
$ neuron-ls
instance-type: trn2.48xlarge
instance-id: i-0dcf246b9a7d5f066
logical-neuroncore-config: 2
+--------+--------+----------+--------+--------------+--------------+---------+---------------+------+---------+
| NEURON | NEURON |  NEURON  | NEURON |  CONNECTED   |     PCI      |   PID   |      CPU      | NUMA | RUNTIME |
| DEVICE | CORES  | CORE IDS | MEMORY |   DEVICES    |     BDF      |         |   AFFINITY    | NODE | VERSION |
+--------+--------+----------+--------+--------------+--------------+---------+---------------+------+---------+
| 0      | 4      | 0-3      | 96 GB  | 12, 1, 9, 14 | 0000:e1:00.0 | 1966680 | 48-95,144-191 | 1    | 2.34.10 |
|        |        |          |        |              |              | 1966681 |               |      | 2.34.10 |
+--------+--------+----------+--------+--------------+--------------+---------+---------------+------+---------+
```

## neuron-ls --json-output

```
$ neuron-ls --json-output
[
    {
        "instance_type": "trn2.48xlarge",
        "instance_id": "i-0dcf246b9a7d5f066",
        "neuron_device": 0,
        "bdf": "0000:e1:00.0",
        "cpu_affinity": "48-95,144-191",
        "numa_node": "1",
        "connected_to": [
            12,
            1,
            9,
            14
        ],
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "memory_size": 103079215104,
        "neuroncore_ids": [
            0,
            1,
            2,
            3
        ],
        "neuron_processes": [
            {
                "pid": 1966680,
                "command": "",
                "neuron_runtime_version": "2.34.10"
            },
            {
                "pid": 1966681,
                "command": "",
                "neuron_runtime_version": "2.34.10"
            }
        ]
    }
]
```

## Neuron env vars

```
$ env | grep -i neuron
NEURON_LOGICAL_NC_CONFIG=2
NEURON_SKIP_EFA_AFFINITY=1
LD_LIBRARY_PATH=/opt/aws/neuron/lib:/opt/amazon/efa/lib:/opt/amazon/efa/lib64:/lib/x86_64-linux-gnu:/opt/conda/lib/
PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/opt/aws/neuron/bin:/opt/conda/bin:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/efa/bin
```

## Python and pip location

```
$ which python3 pip; python3 --version
/opt/conda/bin/python3
/opt/conda/bin/pip
Python 3.13.7
```

## pip list (filtered)

```
$ pip list 2>/dev/null | grep -iE 'neuron|torch|transformers|diffusers|optimum|nki|accelerate|xla|vllm|huggingface|opencv|av'
huggingface_hub                          1.27.0
libtorch-neuronx-lite                    2.11.0.1.0.1284+f49d8626
neuron-agentic-development               1.3
neuronx-cc                               2.27.5334.0+f702b353
nki                                      0.6.0+31049202112.g85070674
opencv-python                            5.0.0.93
opencv-python-headless                   5.0.0.93
torch                                    2.11.0
torch_c_dlpack_ext                       0.1.5
torch-model-archiver                     0.11.0
torch-xla                                2.11.0
torchaudio                               2.11.0
torchserve                               0.11.0
torchvision                              0.26.0
transformers                             5.15.0
vllm                                     0.24.0
vllm-neuron                              0.24.0.1.1.0
```

## torch / torch_neuronx versions

```
$ python3 -c "import torch, torch_neuronx; print(torch.__version__, torch_neuronx.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch, torch_neuronx; print(torch.__version__, torch_neuronx.__version__)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch_neuronx'
[exit code: 1]
```

## neuronx-cc --version

```
$ neuronx-cc --version
NeuronX Compiler version 2.27.5334.0+f702b353

Python version 3.13.7
HWM version 2.27.5334.0+f702b353
NumPy version 2.4.6
```

## import nki

```
$ python3 -c "import nki; print('nki ok')"
nki ok
```

## import neuronxcc.nki

```
$ python3 -c "import neuronxcc.nki; print('neuronxcc.nki ok')"
neuronxcc.nki ok
```

## import optimum.neuron

```
$ python3 -c "import optimum.neuron; print('optimum.neuron ok')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import optimum.neuron; print('optimum.neuron ok')
    ^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'optimum'
[exit code: 1]
```

## diffusers version

```
$ python3 -c "import diffusers; print(diffusers.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import diffusers; print(diffusers.__version__)
    ^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'diffusers'
[exit code: 1]
```

## Neuron tools

```
$ which neuron-top neuron-monitor neuron-profile neuron-explorer
/opt/aws/neuron/bin/neuron-top
/opt/aws/neuron/bin/neuron-monitor
/opt/aws/neuron/bin/neuron-profile
/opt/aws/neuron/bin/neuron-explorer
```

## nproc

```
$ nproc
192
```

## free -h

```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:           2.0Ti        84Gi       1.8Ti       8.5Mi        44Gi       1.9Ti
Swap:             0B          0B          0B
```

## Disk usage

```
$ df -h /workspace /root/.cache/huggingface /dev/shm /tmp
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  2.0T   61G  1.9T   4% /workspace
/dev/nvme0n1p1  2.0T   61G  1.9T   4% /root/.cache/huggingface
tmpfs            16G  180K   16G   1% /dev/shm
overlay         2.0T   61G  1.9T   4% /
```

## HF cache contents (shared across seats on this node)

```
$ ls -la /root/.cache/huggingface/hub
total 4
drwxr-xr-x. 6 root root 134 Oct 10 16:44 .
drwxr-xr-x. 4 root root  57 Oct 10 14:31 ..
drwxr-xr-x. 5 root root 100 Oct 10 16:44 .locks
-rw-r--r--. 1 root root 191 Oct 10 14:31 CACHEDIR.TAG
drwxr-xr-x. 6 root root  65 Oct 10 16:12 models--Qwen--Qwen3-0.6B
drwxr-xr-x. 6 root root  65 Oct 10 16:45 models--Qwen--Qwen3-1.7B
drwxr-xr-x. 6 root root  65 Oct 10 14:32 models--Qwen--Qwen3-8B
```

## vLLM processes (should be empty)

```
$ pgrep -af '[v]llm'
48 /opt/conda/bin/python3.13 /opt/conda/bin/vllm serve --model Qwen/Qwen3-8B --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 4 --block-size 32 --num-gpu-blocks-override 1024 --no-enable-prefix-caching --port 8000
```

## All processes

```
$ ps -eo pid,ppid,etime,rss,cmd --sort=-rss | head -25
    PID    PPID     ELAPSED   RSS CMD
   1411    1085       49:44 3125796 VLLM::Worker_TP1
   1410    1085       49:44 3096004 VLLM::Worker_TP0
     48       1       50:12 1524284 /opt/conda/bin/python3.13 /opt/conda/bin/vllm serve --model Qwen/Qwen3-8B --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 4 --block-size 32 --num-gpu-blocks-override 1024 --no-enable-prefix-caching --port 8000
   1085      48       49:50 1178048 VLLM::EngineCore
  16829   16196       03:43 30564 python -u run_experiment.py --tag s230 --levels levels/06_fifo --runs 1 --rounds 6 --verbose
   1084      48       49:50 13108 /opt/conda/bin/python3.13 -c from multiprocessing.resource_tracker import main;main(33)
  17634   17633       00:00  4148 ps -eo pid,ppid,etime,rss,cmd --sort=-rss
  17156   17154       00:03  4004 bash scripts/env_check.sh
  16196       1       15:43  3616 /bin/bash ./chain_l6.sh
  17633   17632       00:00  3560 bash -c ps -eo pid,ppid,etime,rss,cmd --sort=-rss | head -25
  17631   17156       00:00  2144 bash scripts/env_check.sh
  17632   17631       00:00  2104 timeout 120 bash -c ps -eo pid,ppid,etime,rss,cmd --sort=-rss | head -25
  17154       1       00:03  2064 bash -lc cd /workspace/livevid && nohup setsid bash scripts/env_check.sh > logs/env_check.log 2>&1 < /dev/null & sleep 1; echo started
      1       0    04:02:05  1952 sleep infinity
  17635   17633       00:00  1852 head -25
     44       1       50:12     0 [bash] <defunct>
  14580       1       40:02     0 [python] <defunct>
  15366       1       33:14     0 [python] <defunct>
  15869       1       23:55     0 [chain_l5.sh] <defunct>
  15934       1       21:55     0 [sleep] <defunct>
```

## Outbound: huggingface.co

```
$ curl -sI --max-time 15 https://huggingface.co | head -1
HTTP/2 200 
```

## Outbound: pypi.org

```
$ curl -sI --max-time 15 https://pypi.org | head -1
HTTP/2 200 
```

## uname -a

```
$ uname -a
Linux seat-230 6.18.51-120.162.amzn2023.x86_64 #2 SMP PREEMPT_DYNAMIC Wed Sep 23 18:50:41 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
```

## os-release

```
$ head -3 /etc/os-release
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
```

