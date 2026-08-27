# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
import torch

from protenix.utils import torch_backend


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch_backend.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch_backend.cuda_available():
            # cudnn flags and CUBLAS workspace only apply to CUDA.
            torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic=True applies to CUDA convolution operations, and nothing else.
            torch.backends.cudnn.deterministic = True
            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
