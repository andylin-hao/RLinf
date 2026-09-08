# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import gc
import inspect
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed import distributed_c10d

from rlinf.data.schema.embodied_types import EnvOutput, PolicyOutput
from rlinf.scheduler import (
    Cluster,
    CollectiveGroupOptions,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
    WorkerAddress,
    build_recv_plan,
    build_route_channel_key,
    build_send_plan,
    merge_batches,
    split_batch,
)
from rlinf.scheduler.cluster.config import (
    ClusterConfig,
    CollectiveConfig,
    TensorBufferPoolConfig,
    TensorCompressionConfig,
    TensorCompressionManager,
)
from rlinf.scheduler.collective.collective_group import (
    CollectiveGroup,
    TensorData,
)
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup
from rlinf.scheduler.collective.tensor_buffer_pool import TensorBufferPool
from rlinf.scheduler.collective.tensor_compression import (
    LZ4CodecProvider,
    LZ4CompressionConfig,
    LZ4TensorCodec,
    TensorCompressionWireMetadata,
    ZstdCodecProvider,
    ZstdCompressionConfig,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

SENDER_GROUP_NAME = "sender_worker_group"
RECEIVER_GROUP_NAME = "receiver_worker_group"

# --- Helper Functions ---


def accelerator_is_available():
    """Return whether the Worker accelerator backend is available."""
    return (
        Worker.torch_platform is not None
        and hasattr(Worker.torch_platform, "is_available")
        and Worker.torch_platform.is_available()
    )


def accelerator_device_count():
    """Return accelerator count through the Worker backend abstraction."""
    if Worker.torch_platform is None or not hasattr(
        Worker.torch_platform, "device_count"
    ):
        return 0
    return Worker.torch_platform.device_count()


ACCELERATOR_DEVICE_TYPE = Worker.torch_device_type or "accelerator"


@dataclass
class TensorMessage:
    """Simple dataclass with a tensor field for testing direct tensor send/recv/broadcast."""

    id: int
    payload: torch.Tensor
    note: str


@dataclass
class TensorListMessage:
    """Dataclass with a list of tensors for testing send/recv/broadcast."""

    id: int
    payload_list: list
    note: str


@dataclass
class TensorDictMessage:
    """Dataclass with a dict of tensors for testing send/recv/broadcast."""

    id: int
    payload_dict: dict
    note: str


@dataclass
class PlainMessage:
    """Plain dataclass without tensor fields (sent as Python object)."""

    id: int
    name: str
    value: float


def get_device():
    """Returns the appropriate torch device."""
    if accelerator_is_available():
        Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
        return torch.device(
            f"{Worker.torch_device_type}:{Worker.torch_platform.current_device()}"
        )
    return "cpu"


def get_send_peer_rank(rank, world_size):
    """Calculates the rank of the peer worker."""
    return (rank + 1) % world_size


def get_recv_peer_rank(rank, world_size):
    """Calculates the rank of the peer worker."""
    return (rank - 1) % world_size


NON_CONTIGUOUS_ERR = "must be contiguous when using P2P communication"


def make_non_contiguous_tensor(device):
    """Returns a non-contiguous accelerator tensor (e.g. from .t())."""
    t = torch.ones(2, 3, device=device)
    return t.t()  # transpose is non-contiguous


# --- Worker Definitions ---
class SenderWorker(Worker):
    """Worker responsible for sending data in tests."""

    def __init__(self):
        super().__init__()
        if accelerator_is_available():
            Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))

    def _send_data(self, data, async_op, use_send_tensor=False):
        """Generic data sending method."""
        if accelerator_is_available():
            Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
        peer_rank = get_send_peer_rank(self._rank, self._world_size)
        if use_send_tensor:
            work = self.send_tensor(
                data, RECEIVER_GROUP_NAME, peer_rank, async_op=async_op
            )
        else:
            work = self.send(
                data,
                RECEIVER_GROUP_NAME,
                peer_rank,
                async_op=async_op,
                options=CollectiveGroupOptions(accel_max_ctas=1),
            )

        if async_op:
            work.wait()
        return True

    async def _send_data_asyncio(self, data_factory, use_send_tensor=False):
        """Generic data sending method using asyncio."""

        async def _send():
            if accelerator_is_available():
                Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
            data = data_factory()
            peer_rank = get_send_peer_rank(self._rank, self._world_size)
            if use_send_tensor:
                work = self.send_tensor(
                    data, RECEIVER_GROUP_NAME, peer_rank, async_op=True
                )
            else:
                work = self.send(data, RECEIVER_GROUP_NAME, peer_rank, async_op=True)
            await work.async_wait()
            return True

        return await _send()

    # Sync Tests
    def test_send_object(self, async_op=False):
        return self._send_data({"message": f"Hello from rank {self._rank}"}, async_op)

    def test_send_plain_dataclass(self, async_op=False):
        msg = PlainMessage(
            id=self._rank,
            name=f"rank_{self._rank}",
            value=3.14 * (self._rank + 1),
        )
        return self._send_data(msg, async_op)

    def test_send_tensor(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(2, 2, device=device) * self._rank
        return self._send_data(tensor, async_op)

    def test_send_tensor_list(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor_list = [torch.ones(2, 2, device=device) * i for i in range(4)]
        return self._send_data(tensor_list, async_op)

    def test_send_compressed_data(self, container, async_op=False):
        """Send a compressible CPU tensor in a supported container."""
        tensor = torch.zeros(256 * 1024, dtype=torch.uint8)
        if container == "tensor":
            data = tensor
        elif container == "list":
            data = [tensor, torch.arange(64, dtype=torch.int64)]
        elif container == "tuple":
            data = (tensor, torch.arange(64, dtype=torch.int64))
        elif container == "dict":
            data = {"compressed": tensor, "raw": torch.arange(64)}
        elif container == "dataclass":
            data = TensorMessage(id=self._rank, payload=tensor, note="compressed")
        else:
            raise ValueError(f"Unsupported compressed container: {container}")
        return self._send_data(data, async_op)

    def test_send_tensor_dict(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor_dict = {f"t{i}": torch.ones(2, 2, device=device) * i for i in range(4)}
        return self._send_data(tensor_dict, async_op)

    def test_send_mixed_tensor_list(self, async_op=False):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        cuda_device = get_device()
        tensor_list = [
            torch.ones(2, 2, device="cpu") * (self._rank + 1),
            torch.ones(2, 2, device=cuda_device) * (self._rank + 2),
            torch.ones(2, 2, device="cpu") * (self._rank + 3),
        ]
        return self._send_data(tensor_list, async_op)

    def test_send_mixed_tensor_dict(self, async_op=False):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        cuda_device = get_device()
        tensor_dict = {
            "cpu_a": torch.ones(2, 2, device="cpu") * (self._rank + 1),
            "cuda_b": torch.ones(2, 2, device=cuda_device) * (self._rank + 2),
            "cpu_c": torch.ones(2, 2, device="cpu") * (self._rank + 3),
        }
        return self._send_data(tensor_dict, async_op)

    def test_send_mixed_tensor_list_dataclass(self, async_op=False):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        cuda_device = get_device()
        payload_list = [
            torch.ones(2, 2, device="cpu") * (self._rank + 1),
            torch.ones(2, 2, device=cuda_device) * (self._rank + 2),
            torch.ones(2, 2, device="cpu") * (self._rank + 3),
        ]
        msg = TensorListMessage(
            id=self._rank,
            payload_list=payload_list,
            note=f"mixed list from rank {self._rank}",
        )
        return self._send_data(msg, async_op)

    def test_send_mixed_tensor_dict_dataclass(self, async_op=False):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        cuda_device = get_device()
        payload_dict = {
            "cpu_a": torch.ones(2, 2, device="cpu") * (self._rank + 1),
            "cuda_b": torch.ones(2, 2, device=cuda_device) * (self._rank + 2),
            "cpu_c": torch.ones(2, 2, device="cpu") * (self._rank + 3),
        }
        msg = TensorDictMessage(
            id=self._rank,
            payload_dict=payload_dict,
            note=f"mixed dict from rank {self._rank}",
        )
        return self._send_data(msg, async_op)

    def test_send_tensor_inplace(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(3, 3, device=device) * self._rank
        return self._send_data(tensor, async_op, use_send_tensor=True)

    def test_send_tensor_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * self._rank
        msg = TensorMessage(
            id=self._rank, payload=payload, note=f"from rank {self._rank}"
        )
        return self._send_data(msg, async_op)

    def test_send_tensor_list_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload_list = [
            torch.ones(2, 2, device=device) * (self._rank * 10 + i) for i in range(3)
        ]
        msg = TensorListMessage(
            id=self._rank,
            payload_list=payload_list,
            note=f"list from rank {self._rank}",
        )
        return self._send_data(msg, async_op)

    def test_send_tensor_dict_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload_dict = {
            f"k{i}": torch.ones(2, 2, device=device) * (self._rank * 10 + i)
            for i in range(3)
        }
        msg = TensorDictMessage(
            id=self._rank,
            payload_dict=payload_dict,
            note=f"dict from rank {self._rank}",
        )
        return self._send_data(msg, async_op)

    def test_send_non_contiguous_tensor(self):
        try:
            device = get_device()
            data = make_non_contiguous_tensor(device)
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_list(self):
        try:
            device = get_device()
            data = [make_non_contiguous_tensor(device) for _ in range(2)]
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_dict(self):
        try:
            device = get_device()
            data = {
                "a": make_non_contiguous_tensor(device),
                "b": make_non_contiguous_tensor(device),
            }
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_dataclass(self):
        try:
            device = get_device()
            data = TensorMessage(
                id=1, payload=make_non_contiguous_tensor(device), note="non-contiguous"
            )
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_list_dataclass(self):
        try:
            device = get_device()
            data = TensorListMessage(
                id=1,
                payload_list=[make_non_contiguous_tensor(device) for _ in range(2)],
                note="non-contiguous list",
            )
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_dict_dataclass(self):
        try:
            device = get_device()
            data = TensorDictMessage(
                id=1,
                payload_dict={
                    "a": make_non_contiguous_tensor(device),
                    "b": make_non_contiguous_tensor(device),
                },
                note="non-contiguous dict",
            )
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_tensor_non_contiguous_inplace(self):
        try:
            device = get_device()
            data = make_non_contiguous_tensor(device)
            return self._send_data(data, False, use_send_tensor=True)
        except ValueError as e:
            return e

    # Asyncio Tests
    async def test_send_tensor_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: torch.ones(4, 4, device=device) * self._rank
        )

    async def test_send_tensor_dataclass_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: TensorMessage(
                id=self._rank,
                payload=torch.ones(4, 4, device=device) * self._rank,
                note=f"async from rank {self._rank}",
            )
        )

    async def test_send_tensor_list_dataclass_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: TensorListMessage(
                id=self._rank,
                payload_list=[
                    torch.ones(2, 2, device=device) * (self._rank * 10 + i)
                    for i in range(3)
                ],
                note=f"async list from rank {self._rank}",
            )
        )

    async def test_send_tensor_dict_dataclass_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: TensorDictMessage(
                id=self._rank,
                payload_dict={
                    f"k{i}": torch.ones(2, 2, device=device) * (self._rank * 10 + i)
                    for i in range(3)
                },
                note=f"async dict from rank {self._rank}",
            )
        )

    def test_unaligned_send_recv(self, on_cpu):
        """Test unaligned sending and receiving of tensors."""
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(5, 5, device=device) * self._rank
        peer_rank = get_send_peer_rank(self._rank, self._world_size)
        recv_work = self.recv(RECEIVER_GROUP_NAME, peer_rank, async_op=True)
        self.send(tensor, RECEIVER_GROUP_NAME, peer_rank)
        recv_work.wait()

        recv_tensor = torch.zeros(5, 5, device=device) * self._rank
        recv_work = self.recv_tensor(
            recv_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
        )
        self.send_tensor(tensor, RECEIVER_GROUP_NAME, peer_rank)
        return recv_work.wait()

    def test_consecutive_send_recv(self, on_cpu):
        """Test sending and receiving tensors in a consecutive manner."""
        device = "cpu" if on_cpu else get_device()
        send_tensor = torch.ones(5, 5, device=device) * self._rank
        recv_tensor = torch.zeros(5, 5, device=device)
        send_works = []
        recv_works = []
        peer_rank = get_send_peer_rank(self._rank, self._world_size)
        for _ in range(100):
            send_works.append(
                self.send(send_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True)
            )
            recv_works.append(self.recv(RECEIVER_GROUP_NAME, peer_rank, async_op=True))
            send_works.append(
                self.send_tensor(
                    send_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
                )
            )
            recv_works.append(
                self.recv_tensor(
                    recv_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
                )
            )
        for work in send_works:
            work.wait()
        for work in recv_works:
            work.wait()
        return None

    async def test_memory_leak(self):
        """A test to check for memory leaks during send operations."""
        device = get_device()
        tensor_size = 1024
        large_tensor = torch.randn(tensor_size, dtype=torch.float16, device=device)
        peer_rank = get_send_peer_rank(self._rank, self._world_size)

        self.send(large_tensor, RECEIVER_GROUP_NAME, peer_rank)
        self.send(large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True).wait()
        self.send_tensor(large_tensor, RECEIVER_GROUP_NAME, peer_rank)
        self.send_tensor(
            large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
        ).wait()

        async def _async_send():
            await self.send(
                large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
            ).async_wait()
            await self.send_tensor(
                large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
            ).async_wait()

        await _async_send()

        large_tensor = None
        gc.collect()
        Worker.torch_platform.empty_cache()
        assert Worker.torch_platform.memory_allocated() == 0
        return True


class ReceiverWorker(Worker):
    """Worker responsible for receiving data in tests."""

    def __init__(self):
        super().__init__()
        if accelerator_is_available():
            Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))

    def _recv_data(self, async_op, recv_tensor_inplace_shape=None):
        """Generic data receiving method."""
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        if accelerator_is_available():
            Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
        if recv_tensor_inplace_shape:
            on_cpu, shape = recv_tensor_inplace_shape
            device = "cpu" if on_cpu else get_device()
            tensor = torch.empty(shape, device=device)
            work = self.recv_tensor(
                tensor, SENDER_GROUP_NAME, peer_rank, async_op=async_op
            )
            if async_op:
                work.wait()
            return tensor
        else:
            work = self.recv(
                SENDER_GROUP_NAME,
                peer_rank,
                async_op=async_op,
                options=CollectiveGroupOptions(accel_max_ctas=1),
            )
            if async_op:
                return work.wait()
            return work

    async def _recv_data_asyncio(self, recv_tensor_inplace_shape=None):
        """Generic data receiving method using asyncio."""

        async def _recv():
            if accelerator_is_available():
                Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
            peer_rank = get_recv_peer_rank(self._rank, self._world_size)
            if recv_tensor_inplace_shape:
                on_cpu, shape = recv_tensor_inplace_shape
                device = "cpu" if on_cpu else get_device()
                tensor = torch.empty(shape, device=device)
                work = self.recv_tensor(
                    tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
                )
                await work.async_wait()
                return tensor
            else:
                work = self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True)
                return await work.async_wait()

        return await _recv()

    def test_unaligned_send_recv(self, on_cpu):
        """Test unaligned sending and receiving of tensors."""
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(5, 5, device=device) * self._rank
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        recv_work = self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True)
        self.send(tensor, SENDER_GROUP_NAME, peer_rank)
        recv_work.wait()

        recv_tensor = torch.zeros(5, 5, device=device) * self._rank
        recv_work = self.recv_tensor(
            recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
        )
        self.send_tensor(tensor, SENDER_GROUP_NAME, peer_rank)
        recv_work.wait()
        return recv_tensor

    def test_consecutive_send_recv(self, on_cpu):
        """Test sending and receiving tensors in a consecutive manner."""
        device = "cpu" if on_cpu else get_device()
        send_tensor = torch.ones(5, 5, device=device) * self._rank
        recv_tensor = torch.zeros(5, 5, device=device)
        send_works = []
        recv_works = []
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        for _ in range(100):
            recv_works.append(self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True))
            send_works.append(
                self.send(send_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True)
            )
            recv_works.append(
                self.recv_tensor(
                    recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
                )
            )
            send_works.append(
                self.send_tensor(
                    send_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
                )
            )
        for work in send_works:
            work.wait()
        tensors = [work.wait() for work in recv_works]
        return tensors[0]

    # Sync/Async Wait Tests
    def test_recv_object(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_plain_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_list(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_compressed_data(self, async_op=False):
        """Receive data using the collective compression config."""
        return self._recv_data(async_op)

    def test_recv_tensor_dict(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_inplace(self, on_cpu, async_op=False):
        return self._recv_data(async_op, recv_tensor_inplace_shape=(on_cpu, (3, 3)))

    def test_recv_tensor_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_list_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_dict_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    # Asyncio Tests
    async def test_recv_tensor_asyncio(self, on_cpu):
        return await self._recv_data_asyncio()

    async def test_recv_tensor_dataclass_asyncio(self):
        return await self._recv_data_asyncio()

    async def test_recv_tensor_list_dataclass_asyncio(self):
        return await self._recv_data_asyncio()

    async def test_recv_tensor_dict_dataclass_asyncio(self):
        return await self._recv_data_asyncio()

    async def test_memory_leak(self):
        """A test to check for memory leaks during send operations."""
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        recv_tensor_size = 1024
        device = get_device()
        recv_tensor = torch.randn(recv_tensor_size, dtype=torch.float16, device=device)

        self.recv(SENDER_GROUP_NAME, peer_rank)
        self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True).wait()
        self.recv_tensor(recv_tensor, SENDER_GROUP_NAME, peer_rank)
        self.recv_tensor(
            recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
        ).wait()

        async def _async_recv():
            await self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True).async_wait()
            await self.recv_tensor(
                recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
            ).async_wait()

        await _async_recv()

        recv_tensor = None
        gc.collect()
        Worker.torch_platform.empty_cache()
        assert Worker.torch_platform.memory_allocated() == 0

    async def test_async_wait_yields_control(self):
        """Run recv(async_op=True) and await async_wait() concurrently with another
        asyncio task. Assert the other task ran while waiting, proving async_wait()
        yields control to the event loop."""
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)

        async def recv_task():
            work = self.recv(
                SENDER_GROUP_NAME,
                peer_rank,
                async_op=True,
                options=CollectiveGroupOptions(accel_max_ctas=1),
            )
            return await work.async_wait()

        async def yield_check_task():
            count = 0
            for _ in range(30):
                count += 1
                await asyncio.sleep(0.01)
            return count

        asyncio.create_task(recv_task())
        count = await yield_check_task()
        return count


class CommCollectiveWorker(Worker):
    """Worker for collective communication tests."""

    def __init__(self):
        super().__init__()
        if accelerator_is_available():
            Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))

    def _broadcast_data(self, data, async_op):
        if accelerator_is_available():
            Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
        groups = [(self._group_name, list(range(self._world_size)))]
        payload = data if self._rank == 0 else None
        result = self.broadcast(payload, groups=groups, async_op=async_op)
        if async_op:
            return result.wait()
        return result

    def test_broadcast_object(self, async_op=False):
        payload = {"message": "Hello from rank 0", "rank": 0}
        return self._broadcast_data(payload, async_op)

    def test_broadcast_plain_dataclass(self, async_op=False):
        payload = (
            PlainMessage(id=0, name="broadcast_src", value=2.71)
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * 7
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_list(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = [torch.ones(2, 2, device=device) * i for i in range(4)]
        return self._broadcast_data(payload, async_op)

    def test_broadcast_mixed_tensor_list(self, async_op=False):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        cuda_device = get_device()
        payload = (
            [
                torch.ones(2, 2, device="cpu") * 1,
                torch.ones(2, 2, device=cuda_device) * 2,
                torch.ones(2, 2, device="cpu") * 3,
            ]
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_mixed_tensor_list_dataclass(self, async_op=False):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        cuda_device = get_device()
        payload = (
            TensorListMessage(
                id=0,
                payload_list=[
                    torch.ones(2, 2, device="cpu") * 1,
                    torch.ones(2, 2, device=cuda_device) * 2,
                    torch.ones(2, 2, device="cpu") * 3,
                ],
                note="broadcast mixed list from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_dict(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = {f"t{i}": torch.ones(2, 2, device=device) * i for i in range(4)}
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorMessage(
                id=0,
                payload=torch.ones(2, 2, device=device) * 7,
                note="broadcast from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_list_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorListMessage(
                id=0,
                payload_list=[torch.ones(2, 2, device=device) * i for i in range(4)],
                note="broadcast list from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_dict_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorDictMessage(
                id=0,
                payload_dict={
                    f"t{i}": torch.ones(2, 2, device=device) * i for i in range(4)
                },
                note="broadcast dict from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    async def test_broadcast_tensor_asyncio(self, on_cpu):
        async def _broadcast():
            if accelerator_is_available():
                Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
            device = "cpu" if on_cpu else get_device()
            groups = [(self._group_name, list(range(self._world_size)))]
            payload = torch.ones(3, 3, device=device) * 5
            result = self.broadcast(
                payload if self._rank == 0 else None, groups=groups, async_op=True
            )
            await result.async_wait()
            return result.wait()

        return await _broadcast()

    async def test_cross_group_broadcast_tensor_asyncio(self, groups, on_cpu):
        async def _broadcast():
            if accelerator_is_available():
                Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
            device = "cpu" if on_cpu else get_device()
            src_group_name, src_ranks = groups[0]
            if isinstance(src_ranks, list):
                src_rank = src_ranks[0]
            else:
                src_rank = src_ranks
            is_src = self._worker_address == WorkerAddress(
                src_group_name, ranks=src_rank
            )
            payload = torch.ones(3, 3, device=device) * 9
            result = self.broadcast(
                payload if is_src else None, groups=groups, async_op=True
            )
            await result.async_wait()
            return result.wait()

        return await _broadcast()

    def _cross_group_broadcast(self, groups, payload, async_op):
        src_group_name, src_ranks = groups[0]
        if isinstance(src_ranks, list):
            src_rank = src_ranks[0]
        else:
            src_rank = src_ranks
        is_src = self._worker_address == WorkerAddress(src_group_name, ranks=src_rank)
        result = self.broadcast(
            payload if is_src else None, groups=groups, async_op=async_op
        )
        if async_op:
            return result.wait()
        return result

    def test_cross_group_broadcast_object(self, groups, async_op=False):
        payload = {"message": "Hello from cross-group src", "rank": 0}
        return self._cross_group_broadcast(groups, payload, async_op)

    def test_cross_group_broadcast_tensor(self, groups, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * 11
        return self._cross_group_broadcast(groups, payload, async_op)

    def test_broadcast_object_with_src(self, groups, src, async_op=False):
        payload = {"message": "Hello from explicit src", "rank": 0}
        result = self.broadcast(
            payload if self._worker_address == src else None,
            groups=groups,
            src=(src.root_group_name, src.rank),
            async_op=async_op,
        )
        if async_op:
            return result.wait()
        return result

    def test_broadcast_tensor_with_src(self, groups, src, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * 13
        result = self.broadcast(
            payload if self._worker_address == src else None,
            groups=groups,
            src=(src.root_group_name, src.rank),
            async_op=async_op,
        )
        if async_op:
            return result.wait()
        return result

    def test_broadcast_tensor_dataclass_with_src(
        self, groups, src, on_cpu, async_op=False
    ):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorMessage(
                id=13,
                payload=torch.ones(2, 2, device=device) * 13,
                note="broadcast with src",
            )
            if self._worker_address == src
            else None
        )
        result = self.broadcast(
            payload,
            groups=groups,
            src=(src.root_group_name, src.rank),
            async_op=async_op,
        )
        if async_op:
            return result.wait()
        return result

    def test_cross_group_broadcast_tensor_dataclass(
        self, groups, on_cpu, async_op=False
    ):
        device = "cpu" if on_cpu else get_device()
        src_group_name, src_ranks = groups[0]
        if isinstance(src_ranks, list):
            src_rank = src_ranks[0]
        else:
            src_rank = src_ranks
        is_src = self._worker_address == WorkerAddress(src_group_name, ranks=src_rank)
        payload = (
            TensorMessage(
                id=11,
                payload=torch.ones(2, 2, device=device) * 11,
                note="cross-group broadcast dataclass",
            )
            if is_src
            else None
        )
        return self._cross_group_broadcast(groups, payload, async_op)

    async def test_broadcast_tensor_dataclass_asyncio(self, on_cpu):
        async def _broadcast():
            if accelerator_is_available():
                Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))
            device = "cpu" if on_cpu else get_device()
            groups = [(self._group_name, list(range(self._world_size)))]
            payload = TensorMessage(
                id=5,
                payload=torch.ones(3, 3, device=device) * 5,
                note="async broadcast from rank 0",
            )
            result = self.broadcast(
                payload if self._rank == 0 else None,
                groups=groups,
                async_op=True,
            )
            await result.async_wait()
            return result.wait()

        return await _broadcast()


# --- Pytest Setup ---


@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    return Cluster(
        cluster_cfg=OmegaConf.create(
            {
                "num_nodes": 1,
                "component_placement": [],
                "collective": {
                    "tensor_compression": {
                        "enabled": True,
                        "codec": "lz4",
                        "min_bytes": 1024,
                        "acceleration": 1,
                    }
                },
            }
        )
    )


@pytest.fixture(scope="class")
def worker_groups(cluster: Cluster):
    """Creates and yields the sender and receiver worker groups."""
    if cluster.num_accelerators > 0:
        if cluster.num_accelerators < 4:
            pytest.skip(
                f"NPU send/recv tests require at least 4 accelerator devices, "
                f"found {cluster.num_accelerators}."
            )
        half = cluster.num_accelerators // 2
        sender_placement = PackedPlacementStrategy(0, half - 1)
        receiver_placement = PackedPlacementStrategy(half, cluster.num_accelerators - 1)
        sender_group = SenderWorker.create_group().launch(
            cluster=cluster, placement_strategy=sender_placement, name=SENDER_GROUP_NAME
        )
        receiver_group = ReceiverWorker.create_group().launch(
            cluster=cluster,
            placement_strategy=receiver_placement,
            name=RECEIVER_GROUP_NAME,
        )
    else:
        placement = NodePlacementStrategy([0] * 8)
        sender_group = SenderWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name=SENDER_GROUP_NAME
        )
        receiver_group = ReceiverWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name=RECEIVER_GROUP_NAME
        )
    yield sender_group, receiver_group
    sender_group._close()
    receiver_group._close()


@pytest.fixture(scope="class")
def collective_group(cluster: Cluster):
    """Creates and yields the collective worker group."""
    if cluster.num_accelerators > 0:
        # cross_collective_groups occupies the lower devices (0..cross_size-1) and is
        # alive at the same time as this fixture within TestCollective. Place
        # collective workers on the upper devices to avoid HCCL conflicts.
        cross_size = 4 if cluster.num_accelerators > 4 else 2
        if cluster.num_accelerators <= cross_size:
            pytest.skip(
                f"collective_group requires more than {cross_size} accelerator devices "
                f"so it can run alongside cross_collective_groups without sharing devices. "
                f"Found {cluster.num_accelerators}."
            )
        placement = PackedPlacementStrategy(cross_size, cluster.num_accelerators - 1)
        group = CommCollectiveWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name="collective_group"
        )
    else:
        placement = NodePlacementStrategy([0] * 8)
        group = CommCollectiveWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name="collective_group"
        )
    yield group
    group._close()


@pytest.fixture(scope="class")
def cross_collective_groups(cluster: Cluster):
    """Creates and yields two collective worker groups for cross-group tests."""
    if accelerator_is_available():
        if cluster.num_accelerators < 2:
            pytest.skip("Skipping cross-group tests with insufficient accelerators.")
        if cluster.num_accelerators > 4:
            group_a_size = 2
            group_b_size = 2
        else:
            group_a_size = 1
            group_b_size = 1
        placement_a = PackedPlacementStrategy(0, group_a_size - 1)
        placement_b = PackedPlacementStrategy(
            group_a_size, group_a_size + group_b_size - 1
        )
    else:
        group_a_size = 2
        group_b_size = 2
        placement_a = NodePlacementStrategy([0] * group_a_size)
        placement_b = NodePlacementStrategy([0] * group_b_size)

    group_a = CommCollectiveWorker.create_group().launch(
        cluster=cluster, placement_strategy=placement_a, name="collective_group_a"
    )
    group_b = CommCollectiveWorker.create_group().launch(
        cluster=cluster, placement_strategy=placement_b, name="collective_group_b"
    )
    yield group_a, group_b, group_a_size, group_b_size
    group_a._close()
    group_b._close()


# --- Test Class ---


@pytest.mark.usefixtures("worker_groups")
class TestCommunication:
    """A suite of tests for send/recv communication APIs."""

    def _run_test(
        self,
        worker_groups,
        sender_method,
        receiver_method,
        sender_args=(),
        receiver_args=(),
    ):
        """Helper to run a sender/receiver test pair."""
        sender_group, receiver_group = worker_groups
        sender_results = getattr(sender_group, sender_method)(*sender_args)
        receiver_results = getattr(receiver_group, receiver_method)(*receiver_args)
        results = sender_results.wait()
        results = receiver_results.wait()
        return results

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_object_communication(self, worker_groups, async_op):
        """Tests sending and receiving a Python object."""
        results = self._run_test(
            worker_groups,
            "test_send_object",
            "test_recv_object",
            (async_op,),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert res == {"message": f"Hello from rank {peer_rank}"}

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_plain_dataclass_communication(self, worker_groups, async_op):
        """Tests sending and receiving a plain dataclass without tensor fields."""
        results = self._run_test(
            worker_groups,
            "test_send_plain_dataclass",
            "test_recv_plain_dataclass",
            (async_op,),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, PlainMessage)
            assert res.id == peer_rank
            assert res.name == f"rank_{peer_rank}"
            assert res.value == pytest.approx(3.14 * (peer_rank + 1))

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a single tensor."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor",
            "test_recv_tensor",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected = torch.ones(2, 2) * peer_rank
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_list_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a list of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_list",
            "test_recv_tensor_list",
            (on_cpu, async_op),
            (async_op,),
        )
        for res_list in results:
            assert isinstance(res_list, list)
            for i, tensor in enumerate(res_list):
                expected = torch.ones(2, 2) * i
                assert torch.equal(tensor.cpu(), expected)

    @pytest.mark.parametrize(
        "container", ["tensor", "list", "tuple", "dict", "dataclass"]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_compressed_cpu_tensor_communication(
        self, worker_groups, container, async_op
    ):
        """Compressed CPU tensors preserve values on supported container paths."""
        results = self._run_test(
            worker_groups,
            "test_send_compressed_data",
            "test_recv_compressed_data",
            (container, async_op),
            (async_op,),
        )
        expected = torch.zeros(256 * 1024, dtype=torch.uint8)
        for result in results:
            if container == "tensor":
                tensor = result
            elif container in {"list", "tuple"}:
                # Tuple inputs use the established tensor-list wire path.
                assert isinstance(result, list)
                tensor = result[0]
                assert torch.equal(result[1], torch.arange(64, dtype=torch.int64))
            elif container == "dict":
                tensor = result["compressed"]
                assert torch.equal(result["raw"], torch.arange(64))
            else:
                assert isinstance(result, TensorMessage)
                assert result.note == "compressed"
                tensor = result.payload
            assert torch.equal(tensor, expected)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_mixed_tensor_list_communication(self, worker_groups, async_op):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_mixed_tensor_list",
            "test_recv_tensor_list",
            (async_op,),
            (async_op,),
        )
        for i, res_list in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected_vals = [peer_rank + 1, peer_rank + 2, peer_rank + 3]
            expected_devices = ["cpu", Worker.torch_device_type, "cpu"]
            for tensor, expected_val, expected_device in zip(
                res_list, expected_vals, expected_devices
            ):
                assert tensor.device.type == expected_device
                assert torch.equal(tensor.cpu(), torch.ones(2, 2) * expected_val)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_mixed_tensor_dict_communication(self, worker_groups, async_op):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_mixed_tensor_dict",
            "test_recv_tensor_dict",
            (async_op,),
            (async_op,),
        )
        for i, res_dict in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert res_dict["cpu_a"].device.type == "cpu"
            assert res_dict["cuda_b"].device.type == Worker.torch_device_type
            assert res_dict["cpu_c"].device.type == "cpu"
            assert torch.equal(
                res_dict["cpu_a"].cpu(), torch.ones(2, 2) * (peer_rank + 1)
            )
            assert torch.equal(
                res_dict["cuda_b"].cpu(), torch.ones(2, 2) * (peer_rank + 2)
            )
            assert torch.equal(
                res_dict["cpu_c"].cpu(), torch.ones(2, 2) * (peer_rank + 3)
            )

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_mixed_tensor_list_dataclass_communication(self, worker_groups, async_op):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_mixed_tensor_list_dataclass",
            "test_recv_tensor_list_dataclass",
            (async_op,),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorListMessage)
            assert res.id == peer_rank
            assert res.note == f"mixed list from rank {peer_rank}"
            expected_vals = [peer_rank + 1, peer_rank + 2, peer_rank + 3]
            expected_devices = ["cpu", Worker.torch_device_type, "cpu"]
            for tensor, expected_val, expected_device in zip(
                res.payload_list, expected_vals, expected_devices
            ):
                assert tensor.device.type == expected_device
                assert torch.equal(tensor.cpu(), torch.ones(2, 2) * expected_val)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_mixed_tensor_dict_dataclass_communication(self, worker_groups, async_op):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_mixed_tensor_dict_dataclass",
            "test_recv_tensor_dict_dataclass",
            (async_op,),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorDictMessage)
            assert res.id == peer_rank
            assert res.note == f"mixed dict from rank {peer_rank}"
            assert res.payload_dict["cpu_a"].device.type == "cpu"
            assert res.payload_dict["cuda_b"].device.type == Worker.torch_device_type
            assert res.payload_dict["cpu_c"].device.type == "cpu"
            assert torch.equal(
                res.payload_dict["cpu_a"].cpu(), torch.ones(2, 2) * (peer_rank + 1)
            )
            assert torch.equal(
                res.payload_dict["cuda_b"].cpu(), torch.ones(2, 2) * (peer_rank + 2)
            )
            assert torch.equal(
                res.payload_dict["cpu_c"].cpu(), torch.ones(2, 2) * (peer_rank + 3)
            )

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_dict_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dictionary of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dict",
            "test_recv_tensor_dict",
            (on_cpu, async_op),
            (async_op,),
        )
        for res_dict in results:
            assert isinstance(res_dict, dict)
            for i, key in enumerate(sorted(res_dict.keys())):
                assert key == f"t{i}"
                expected = torch.ones(2, 2) * i
                assert torch.equal(res_dict[key].cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_inplace_tensor_communication(self, worker_groups, on_cpu, async_op):
        """Tests send_tensor/recv_tensor for in-place tensor communication."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_inplace",
            "test_recv_tensor_inplace",
            (on_cpu, async_op),
            (on_cpu, async_op),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected = torch.ones(3, 3) * peer_rank
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_dataclass_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dataclass containing torch tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dataclass",
            "test_recv_tensor_dataclass",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorMessage)
            assert res.id == peer_rank
            assert res.note == f"from rank {peer_rank}"
            assert torch.equal(res.payload.cpu(), torch.ones(2, 2) * peer_rank)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_list_dataclass_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dataclass containing a list of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_list_dataclass",
            "test_recv_tensor_list_dataclass",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorListMessage)
            assert res.id == peer_rank
            assert res.note == f"list from rank {peer_rank}"
            assert len(res.payload_list) == 3
            for j, t in enumerate(res.payload_list):
                expected = torch.ones(2, 2) * (peer_rank * 10 + j)
                assert torch.equal(t.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_dict_dataclass_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dataclass containing a dict of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dict_dataclass",
            "test_recv_tensor_dict_dataclass",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorDictMessage)
            assert res.id == peer_rank
            assert res.note == f"dict from rank {peer_rank}"
            assert list(res.payload_dict.keys()) == ["k0", "k1", "k2"]
            for j, key in enumerate(sorted(res.payload_dict.keys())):
                expected = torch.ones(2, 2) * (peer_rank * 10 + j)
                assert torch.equal(res.payload_dict[key].cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async communication with asyncio.run and async_wait."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_asyncio",
            "test_recv_tensor_asyncio",
            (on_cpu,),
            (on_cpu,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected = torch.ones(4, 4) * peer_rank
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_tensor_dataclass_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async send/recv of dataclass containing torch tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dataclass_asyncio",
            "test_recv_tensor_dataclass_asyncio",
            (on_cpu,),
            (),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorMessage)
            assert res.id == peer_rank
            assert res.note == f"async from rank {peer_rank}"
            assert torch.equal(res.payload.cpu(), torch.ones(4, 4) * peer_rank)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_tensor_list_dataclass_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async send/recv of dataclass containing list of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_list_dataclass_asyncio",
            "test_recv_tensor_list_dataclass_asyncio",
            (on_cpu,),
            (),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorListMessage)
            assert res.id == peer_rank
            assert res.note == f"async list from rank {peer_rank}"
            assert len(res.payload_list) == 3
            for j, t in enumerate(res.payload_list):
                assert torch.equal(t.cpu(), torch.ones(2, 2) * (peer_rank * 10 + j))

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_tensor_dict_dataclass_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async send/recv of dataclass containing dict of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dict_dataclass_asyncio",
            "test_recv_tensor_dict_dataclass_asyncio",
            (on_cpu,),
            (),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorDictMessage)
            assert res.id == peer_rank
            assert res.note == f"async dict from rank {peer_rank}"
            for j, key in enumerate(sorted(res.payload_dict.keys())):
                assert torch.equal(
                    res.payload_dict[key].cpu(),
                    torch.ones(2, 2) * (peer_rank * 10 + j),
                )

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_unaligned_send_recv(self, worker_groups, on_cpu):
        """Tests unaligned sending and receiving of tensors."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_unaligned_send_recv",
            "test_unaligned_send_recv",
            (on_cpu,),
            (on_cpu,),
        )
        for i, res in enumerate(results):
            expected = torch.ones(5, 5) * get_recv_peer_rank(i, len(results))
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_consecutive_send_recv(self, worker_groups, on_cpu):
        """Tests sending and receiving tensors in a consecutive manner."""
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_test(
            worker_groups,
            "test_consecutive_send_recv",
            "test_consecutive_send_recv",
            (on_cpu,),
            (on_cpu,),
        )
        for i, res in enumerate(results):
            expected = torch.ones(5, 5) * get_recv_peer_rank(i, len(results))
            assert torch.equal(res.cpu(), expected)

    def test_memory_leak(self, worker_groups):
        """Tests unaligned sending and receiving of tensors."""
        if not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        self._run_test(
            worker_groups,
            "test_memory_leak",
            "test_memory_leak",
        )

    def test_async_wait_yields_control(self, worker_groups):
        """Ensures async_wait() of async comm correctly yields control so other
        asyncio tasks can run while waiting."""
        sender_group, receiver_group = worker_groups
        # Run on rank 0 only to avoid multi-worker timing; receiver waits, sender sends after delay.
        recv_ref = receiver_group.execute_on(1).test_async_wait_yields_control()

        def delayed_send():
            time.sleep(0.15)
            sender_group.execute_on(0).test_send_object(False).wait()

        t = threading.Thread(target=delayed_send)
        try:
            results = recv_ref.wait()
            t.start()
        finally:
            t.join()
        for i, yield_count in enumerate(results):
            assert yield_count >= 1, (
                f"async_wait() did not yield: yield_check task ran {yield_count} times"
            )

    @pytest.mark.parametrize(
        "sender_method",
        [
            "test_send_non_contiguous_tensor",
            "test_send_non_contiguous_tensor_list",
            "test_send_non_contiguous_tensor_dict",
            "test_send_non_contiguous_tensor_dataclass",
            "test_send_non_contiguous_tensor_list_dataclass",
            "test_send_non_contiguous_tensor_dict_dataclass",
            "test_send_tensor_non_contiguous_inplace",
        ],
    )
    def test_non_contiguous_send_raises_value_error(self, worker_groups, sender_method):
        """Sending non-contiguous accelerator tensors (any struct) must raise ValueError."""
        if not accelerator_is_available():
            pytest.skip("Skipping non-contiguous tests on CPU-only environment.")
        sender_group, _ = worker_groups
        results = getattr(sender_group.execute_on(0), sender_method)().wait()
        err = results[0]
        assert isinstance(err, ValueError), (
            f"Expected ValueError, got {type(err)}: {err}"
        )
        assert NON_CONTIGUOUS_ERR in str(err), (
            f"Expected message containing {NON_CONTIGUOUS_ERR!r}, got: {err}"
        )


@pytest.mark.usefixtures("collective_group")
class TestCollective:
    """A suite of tests for collective communication APIs."""

    def _run_collective_test(self, collective_group, method, *args):
        results = getattr(collective_group, method)(*args).wait()
        return results

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_object(self, collective_group, async_op):
        results = self._run_collective_test(
            collective_group, "test_broadcast_object", async_op
        )
        for res in results:
            assert res == {"message": "Hello from rank 0", "rank": 0}

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_plain_dataclass(self, collective_group, async_op):
        """Tests broadcasting a plain dataclass without tensor fields."""
        results = self._run_collective_test(
            collective_group, "test_broadcast_plain_dataclass", async_op
        )
        for res in results:
            assert isinstance(res, PlainMessage)
            assert res.id == 0
            assert res.name == "broadcast_src"
            assert res.value == pytest.approx(2.71)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor(self, collective_group, on_cpu, async_op):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor", on_cpu, async_op
        )
        expected = torch.ones(2, 2) * 7
        for res in results:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_list(self, collective_group, on_cpu, async_op):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_list", on_cpu, async_op
        )
        for res_list in results:
            assert isinstance(res_list, list)
            for i, tensor in enumerate(res_list):
                expected = torch.ones(2, 2) * i
                assert torch.equal(tensor.cpu(), expected)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_mixed_tensor_list(self, collective_group, async_op):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_mixed_tensor_list", async_op
        )
        expected_vals = [1, 2, 3]
        expected_devices = ["cpu", Worker.torch_device_type, "cpu"]
        for res_list in results:
            for tensor, expected_val, expected_device in zip(
                res_list, expected_vals, expected_devices
            ):
                assert tensor.device.type == expected_device
                assert torch.equal(tensor.cpu(), torch.ones(2, 2) * expected_val)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_mixed_tensor_list_dataclass(self, collective_group, async_op):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_mixed_tensor_list_dataclass", async_op
        )
        expected_vals = [1, 2, 3]
        expected_devices = ["cpu", Worker.torch_device_type, "cpu"]
        for res in results:
            assert isinstance(res, TensorListMessage)
            assert res.id == 0
            assert res.note == "broadcast mixed list from rank 0"
            for tensor, expected_val, expected_device in zip(
                res.payload_list, expected_vals, expected_devices
            ):
                assert tensor.device.type == expected_device
                assert torch.equal(tensor.cpu(), torch.ones(2, 2) * expected_val)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dict(self, collective_group, on_cpu, async_op):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_dict", on_cpu, async_op
        )
        for res_dict in results:
            assert isinstance(res_dict, dict)
            for i, key in enumerate(sorted(res_dict.keys())):
                assert key == f"t{i}"
                expected = torch.ones(2, 2) * i
                assert torch.equal(res_dict[key].cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dataclass(self, collective_group, on_cpu, async_op):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_dataclass", on_cpu, async_op
        )
        expected_payload = torch.ones(2, 2) * 7
        for res in results:
            assert isinstance(res, TensorMessage)
            assert res.id == 0
            assert res.note == "broadcast from rank 0"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_list_dataclass(self, collective_group, on_cpu, async_op):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group,
            "test_broadcast_tensor_list_dataclass",
            on_cpu,
            async_op,
        )
        for res in results:
            assert isinstance(res, TensorListMessage)
            assert res.id == 0
            assert res.note == "broadcast list from rank 0"
            assert len(res.payload_list) == 4
            for i, t in enumerate(res.payload_list):
                assert torch.equal(t.cpu(), torch.ones(2, 2) * i)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dict_dataclass(self, collective_group, on_cpu, async_op):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group,
            "test_broadcast_tensor_dict_dataclass",
            on_cpu,
            async_op,
        )
        for res in results:
            assert isinstance(res, TensorDictMessage)
            assert res.id == 0
            assert res.note == "broadcast dict from rank 0"
            assert list(res.payload_dict.keys()) == ["t0", "t1", "t2", "t3"]
            for i, key in enumerate(sorted(res.payload_dict.keys())):
                assert torch.equal(res.payload_dict[key].cpu(), torch.ones(2, 2) * i)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_cross_group_broadcast_object(self, cross_collective_groups, async_op):
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_object(groups, async_op)
        handle_b = group_b.test_cross_group_broadcast_object(groups, async_op)
        results = handle_a.wait() + handle_b.wait()
        for res in results:
            assert res == {"message": "Hello from cross-group src", "rank": 0}

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_cross_group_broadcast_tensor(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_tensor(groups, on_cpu, async_op)
        handle_b = group_b.test_cross_group_broadcast_tensor(groups, on_cpu, async_op)
        results = handle_a.wait() + handle_b.wait()
        expected = torch.ones(2, 2) * 11
        for res in results:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_cross_group_broadcast_tensor_dataclass(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_tensor_dataclass(
            groups, on_cpu, async_op
        )
        handle_b = group_b.test_cross_group_broadcast_tensor_dataclass(
            groups, on_cpu, async_op
        )
        results = handle_a.wait() + handle_b.wait()
        expected_payload = torch.ones(2, 2) * 11
        for res in results:
            assert isinstance(res, TensorMessage)
            assert res.id == 11
            assert res.note == "cross-group broadcast dataclass"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_src_ignores_group_order(self, cross_collective_groups, async_op):
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups_a_first = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        groups_b_first = [
            ("collective_group_b", list(range(group_b_size))),
            ("collective_group_a", list(range(group_a_size))),
        ]
        src_addr = WorkerAddress("collective_group_a", ranks=0)
        handle_a = group_a.test_broadcast_object_with_src(
            groups_a_first, src_addr, async_op
        )
        handle_b = group_b.test_broadcast_object_with_src(
            groups_a_first, src_addr, async_op
        )
        results_a = handle_a.wait() + handle_b.wait()
        handle_a = group_a.test_broadcast_object_with_src(
            groups_b_first, src_addr, async_op
        )
        handle_b = group_b.test_broadcast_object_with_src(
            groups_b_first, src_addr, async_op
        )
        results_b = handle_a.wait() + handle_b.wait()
        assert results_a == results_b

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_src_tensor_order_independent(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups_a_first = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        groups_b_first = [
            ("collective_group_b", list(range(group_b_size))),
            ("collective_group_a", list(range(group_a_size))),
        ]
        src_addr = WorkerAddress("collective_group_a", ranks=0)
        handle_a = group_a.test_broadcast_tensor_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        results_a = handle_a.wait() + handle_b.wait()
        handle_a = group_a.test_broadcast_tensor_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        results_b = handle_a.wait() + handle_b.wait()
        expected = torch.ones(2, 2) * 13
        for res in results_a + results_b:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dataclass_with_src(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups_a_first = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        groups_b_first = [
            ("collective_group_b", list(range(group_b_size))),
            ("collective_group_a", list(range(group_a_size))),
        ]
        src_addr = WorkerAddress("collective_group_a", ranks=0)
        handle_a = group_a.test_broadcast_tensor_dataclass_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_dataclass_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        results_a = handle_a.wait() + handle_b.wait()
        handle_a = group_a.test_broadcast_tensor_dataclass_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_dataclass_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        results_b = handle_a.wait() + handle_b.wait()
        expected_payload = torch.ones(2, 2) * 13
        for res in results_a + results_b:
            assert isinstance(res, TensorMessage)
            assert res.id == 13
            assert res.note == "broadcast with src"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_collective_asyncio_broadcast(self, collective_group, on_cpu):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_asyncio", on_cpu
        )
        expected = torch.ones(3, 3) * 5
        for res in results:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_collective_asyncio_broadcast_tensor_dataclass(
        self, collective_group, on_cpu
    ):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_dataclass_asyncio", on_cpu
        )
        expected_payload = torch.ones(3, 3) * 5
        for res in results:
            assert isinstance(res, TensorMessage)
            assert res.id == 5
            assert res.note == "async broadcast from rank 0"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize(
        "on_cpu", [True, False], ids=["cpu", ACCELERATOR_DEVICE_TYPE]
    )
    def test_collective_asyncio_cross_group_broadcast(
        self, cross_collective_groups, on_cpu
    ):
        if not on_cpu and not accelerator_is_available():
            pytest.skip("Skipping accelerator test without an accelerator.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_tensor_asyncio(groups, on_cpu)
        handle_b = group_b.test_cross_group_broadcast_tensor_asyncio(groups, on_cpu)
        results = handle_a.wait() + handle_b.wait()
        expected = torch.ones(3, 3) * 9
        for res in results:
            assert torch.equal(res.cpu(), expected)


class _BroadcastFailureGroup:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def broadcast(self, tensors: list[torch.Tensor], options: object) -> None:
        del tensors, options
        raise self._error

    def __repr__(self) -> str:
        return "_BroadcastFailureGroup()"


class _WaitFailureWork:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def wait(self) -> None:
        raise self._error


class _WaitFailureGroup:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def broadcast(
        self, tensors: list[torch.Tensor], options: object
    ) -> _WaitFailureWork:
        del tensors, options
        return _WaitFailureWork(self._error)

    def __repr__(self) -> str:
        return "_WaitFailureGroup()"


@pytest.fixture
def multi_channel_group(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> MultiChannelProcessGroup:
    monkeypatch.setattr(distributed_c10d, "BroadcastOptions", SimpleNamespace)
    monkeypatch.setattr(
        distributed_c10d, "_check_single_tensor", lambda tensor, name: None
    )
    monkeypatch.setattr(distributed_c10d, "_rank_not_in_group", lambda group: False)
    monkeypatch.setattr(distributed_c10d, "get_group_rank", lambda group, rank: rank)
    monkeypatch.setattr(dist, "_get_process_group_name", lambda group: "test-group")

    logger = logging.getLogger(__name__)
    caplog.set_level(logging.ERROR, logger=logger.name)
    process_group = object.__new__(MultiChannelProcessGroup)
    process_group._cur_rank = 1
    process_group._peer_rank = 0
    process_group._num_channels = 1
    process_group._is_initialized = True
    process_group._no_accel_ccl = False
    process_group._logger = logger
    return process_group


class TestMultiChannelProcessGroupFailures:
    """Tests that broadcast failures propagate out of the receive path."""

    @staticmethod
    def _assert_failure_log(
        caplog: pytest.LogCaptureFixture, expected_error: str
    ) -> None:
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "ProcessGroup test-group rank 1" in message
        assert expected_error in message

    def test_recv_propagates_process_group_failure(
        self,
        multi_channel_group: MultiChannelProcessGroup,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        error = RuntimeError("connection closed by peer")
        multi_channel_group._recv_gloo_process_groups = [_BroadcastFailureGroup(error)]

        with pytest.raises(RuntimeError) as exc_info:
            multi_channel_group.recv(
                torch.empty(1),
                device=CollectiveGroup.CPU,
                channel_id=0,
            )

        assert exc_info.value is error
        self._assert_failure_log(caplog, str(error))

    def test_recv_propagates_synchronous_wait_failure(
        self,
        multi_channel_group: MultiChannelProcessGroup,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        error = RuntimeError("timed out waiting for recv")
        multi_channel_group._recv_gloo_process_groups = [_WaitFailureGroup(error)]

        with pytest.raises(RuntimeError) as exc_info:
            multi_channel_group.recv(
                torch.empty(1),
                device=CollectiveGroup.CPU,
                channel_id=0,
            )

        assert exc_info.value is error
        self._assert_failure_log(caplog, str(error))


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def _make_obs(start: int, batch_size: int) -> dict:
    return {
        "states": torch.arange(start, start + batch_size * 2, dtype=torch.float32).view(
            batch_size, 2
        ),
        "main_images": None,
        "wrist_images": None,
        "extra_view_images": None,
        "task_descriptions": [
            f"task-{idx}" for idx in range(start, start + batch_size)
        ],
    }


def test_build_send_plan_load_balance_env_to_rollout():
    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=0,
        src_world_size=2,
        dst_world_size=3,
        tag="train_obs",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (0, 4),
        (1, 2),
    ]

    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=1,
        src_world_size=2,
        dst_world_size=3,
        tag="train_obs",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (1, 2),
        (2, 4),
    ]


def test_build_send_plan_load_balance_rollout_to_env():
    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=0,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(0, 4)]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=1,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (0, 2),
        (1, 2),
    ]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=2,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(1, 4)]


def test_build_recv_plan_matches_expected_receive_sizes():
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=0,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(0, 4)]
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=1,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(0, 2), (1, 2)]
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=2,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(1, 4)]


def test_build_route_channel_key_is_stable():
    assert build_route_channel_key("env", "rollout", 2, 1, "train") == (
        "scheduler_route",
        "env",
        "rollout",
        "train",
        "",
        2,
        1,
    )
    assert build_route_channel_key("rollout", "env", 0, 3, "eval", "k") == (
        "scheduler_route",
        "rollout",
        "env",
        "eval",
        "k",
        0,
        3,
    )


def test_split_and_merge_nested_batches():
    batch = {
        "obs": _make_obs(0, 6),
        "final_obs": None,
        "rewards": torch.arange(6, dtype=torch.float32).unsqueeze(-1),
    }
    shards = split_batch(batch, [4, 2])
    assert shards[0]["obs"]["states"].shape[0] == 4
    assert len(shards[1]["obs"]["task_descriptions"]) == 2

    merged = merge_batches(shards)
    assert torch.equal(merged["obs"]["states"], batch["obs"]["states"])
    assert merged["obs"]["task_descriptions"] == batch["obs"]["task_descriptions"]
    assert torch.equal(merged["rewards"], batch["rewards"])


def test_policy_output_split_merge_invariant():
    policy_output = PolicyOutput(
        actions=torch.arange(12, dtype=torch.float32).view(6, 2),
        prev_logprobs=torch.arange(12, dtype=torch.float32).view(6, 2),
        prev_values=torch.arange(6, dtype=torch.float32).view(6, 1),
        bootstrap_values=torch.arange(6, dtype=torch.float32).view(6, 1),
        intervene_flags=torch.ones((6, 3), dtype=torch.bool),
        forward_inputs={
            "action": torch.arange(12, dtype=torch.float32).view(6, 2),
            "states": torch.arange(18, dtype=torch.float32).view(6, 3),
        },
        versions=torch.arange(6, dtype=torch.float32).view(6, 1),
    )

    worker = object.__new__(MultiStepRolloutWorker)
    shards = worker._split_policy_output(policy_output, [4, 2])
    merged = PolicyOutput.merge(shards)

    assert torch.equal(merged.actions, policy_output.actions)
    assert torch.equal(merged.prev_logprobs, policy_output.prev_logprobs)
    assert torch.equal(merged.prev_values, policy_output.prev_values)
    assert torch.equal(merged.bootstrap_values, policy_output.bootstrap_values)
    assert torch.equal(merged.intervene_flags, policy_output.intervene_flags)
    assert torch.equal(
        merged.forward_inputs["action"], policy_output.forward_inputs["action"]
    )
    assert torch.equal(
        merged.forward_inputs["states"], policy_output.forward_inputs["states"]
    )
    assert torch.equal(merged.versions, policy_output.versions)


def test_merge_env_outputs_with_partial_optional_fields():
    env_output_0 = EnvOutput(
        obs=_make_obs(0, 2),
        final_obs=None,
        dones=torch.zeros((2, 1), dtype=torch.bool),
        terminations=torch.zeros((2, 1), dtype=torch.bool),
        truncations=torch.zeros((2, 1), dtype=torch.bool),
        rewards=torch.ones((2, 1), dtype=torch.float32),
        intervene_actions=None,
        intervene_flags=None,
    ).to_dict()
    env_output_1 = EnvOutput(
        obs=_make_obs(100, 3),
        final_obs=_make_obs(200, 3),
        dones=torch.zeros((3, 1), dtype=torch.bool),
        terminations=torch.zeros((3, 1), dtype=torch.bool),
        truncations=torch.zeros((3, 1), dtype=torch.bool),
        rewards=torch.ones((3, 1), dtype=torch.float32) * 2,
        intervene_actions=torch.ones((3, 4), dtype=torch.float32),
        intervene_flags=torch.ones((3, 1), dtype=torch.bool),
        rlt_switch_flags=torch.ones((3, 1), dtype=torch.bool),
    ).to_dict()

    merged = EnvOutput.merge_env_outputs([env_output_0, env_output_1])

    assert merged["obs"]["states"].shape[0] == 5
    assert len(merged["obs"]["task_descriptions"]) == 5
    assert merged["rewards"].shape[0] == 5
    assert merged["final_obs"] is not None
    assert torch.equal(merged["final_obs"]["states"][:2], env_output_0["obs"]["states"])
    assert torch.equal(
        merged["final_obs"]["states"][2:], env_output_1["final_obs"]["states"]
    )

    assert merged["intervene_actions"].shape == (5, 4)
    assert torch.equal(
        merged["intervene_actions"][:2], torch.zeros((2, 4), dtype=torch.float32)
    )
    assert merged["intervene_flags"].shape == (5, 1)
    assert torch.equal(
        merged["intervene_flags"][:2], torch.zeros((2, 1), dtype=torch.bool)
    )
    assert merged["rlt_switch_flags"].shape == (5, 1)
    assert torch.equal(
        merged["rlt_switch_flags"][:2], torch.zeros((2, 1), dtype=torch.bool)
    )


def test_buffer_pool_uses_best_fit_buffers_independent_of_tensor_order():
    """The Worker-wide pool chooses the smallest tensor buffer that fits."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=1024))
    large_lease = pool.try_acquire(512)
    small_lease = pool.try_acquire(128)
    assert large_lease is not None
    assert small_lease is not None
    large_buffer = large_lease.tensor
    small_buffer = small_lease.tensor
    large_lease.release()
    small_lease.release()

    small_reuse = pool.try_acquire(100)
    large_reuse = pool.try_acquire(300)
    assert small_reuse is not None
    assert large_reuse is not None
    assert small_reuse.tensor.data_ptr() == small_buffer.data_ptr()
    assert large_reuse.tensor.data_ptr() == large_buffer.data_ptr()
    small_reuse.release()
    large_reuse.release()


def test_buffer_pool_reuses_same_size_bucket_and_tracks_cached_bytes():
    """Equal-sized buffers share a bucket and update cached accounting eagerly."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=1024))
    leases = [pool.try_acquire(128), pool.try_acquire(128), pool.try_acquire(256)]
    assert all(lease is not None for lease in leases)
    pointers = {lease.tensor.data_ptr() for lease in leases}

    for lease in leases:
        lease.release()
    assert pool.cached_bytes == 512

    first = pool.try_acquire(100)
    second = pool.try_acquire(200)
    assert first is not None
    assert second is not None
    assert first.tensor.data_ptr() in pointers
    assert second.tensor.data_ptr() in pointers
    assert pool.cached_bytes == 128
    first.release(cache=False)
    second.release(cache=False)


def test_buffer_pool_never_exceeds_its_worker_budget():
    """Active buffers make later acquisitions fall back without overallocating."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=512))
    buffer = pool.try_acquire(400)
    assert buffer is not None
    assert pool.try_acquire(200) is None
    assert pool.allocated_bytes == 400

    buffer.release(cache=False)
    assert pool.allocated_bytes == 0


def test_buffer_pool_evicts_idle_buffers_to_fit_a_new_shape():
    """Historical shapes cannot make the bounded cache grow indefinitely."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=512))
    old_buffer = pool.try_acquire(128)
    assert old_buffer is not None
    old_buffer.release()

    replacement = pool.try_acquire(512)
    assert replacement is not None
    assert pool.allocated_bytes == 512
    replacement.release()


def test_buffer_pool_evicts_an_entire_size_bucket_in_one_step():
    """A new large allocation can replace many equal-sized idle buffers."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=512))
    leases = [pool.try_acquire(128) for _ in range(4)]
    assert all(lease is not None for lease in leases)
    for lease in leases:
        lease.release()

    replacement = pool.try_acquire(512)
    assert replacement is not None
    assert pool.allocated_bytes == 512
    assert pool.cached_bytes == 0
    replacement.release()


def test_buffer_pool_preserves_a_large_buffer_when_a_small_one_can_fit():
    """A speculative small compression cannot consume a much larger buffer."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=256))
    large_lease = pool.try_acquire(128)
    assert large_lease is not None
    large_buffer = large_lease.tensor
    large_lease.release()

    small_lease = pool.try_acquire(1)
    assert small_lease is not None
    assert small_lease.tensor.data_ptr() != large_buffer.data_ptr()
    small_lease.release(cache=False)
    reused_large = pool.try_acquire(100)
    assert reused_large is not None
    assert reused_large.tensor.data_ptr() == large_buffer.data_ptr()
    reused_large.release()


def test_zstd_provider_never_waits_for_a_busy_compressor():
    """Saturated Zstd context pools do not block a sender."""
    provider = ZstdCodecProvider(ZstdCompressionConfig(max_inflight=1))

    codec = provider.try_acquire_compressor()
    assert codec is not None
    assert provider.try_acquire_compressor() is None
    provider.release(codec)

    reused_codec = provider.try_acquire_compressor()
    assert reused_codec is not None
    provider.release(reused_codec)


def test_lz4_provider_supports_concurrent_round_trips():
    """The shared stateless LZ4 instance is safe across Worker threads."""
    provider = LZ4CodecProvider(LZ4CompressionConfig())

    def round_trip(value: int) -> bool:
        source = torch.full((128 * 1024,), value, dtype=torch.uint8)
        compressor = provider.try_acquire_compressor()
        assert compressor is not None
        try:
            capacity = compressor.compress_bound(source.numel())
            assert capacity is not None
            compressed = torch.empty(capacity, dtype=torch.uint8)
            compressed_numel = compressor.compress_into(source, compressed)
        finally:
            provider.release(compressor)

        restored = torch.empty_like(source)
        decompressor = provider.acquire_decompressor()
        try:
            decompressor.decompress_into(compressed, compressed_numel, restored)
        finally:
            provider.release(decompressor)
        return torch.equal(restored, source)

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(round_trip, range(16)))


@pytest.mark.parametrize("codec", ["lz4", "zstd"])
def test_codec_provider_compresses_and_restores_a_tensor(codec):
    """A provider's codec writes and restores tensor bytes."""
    compression_config = (
        LZ4CompressionConfig() if codec == "lz4" else ZstdCompressionConfig()
    )
    codec_provider = compression_config.create_codec_provider()
    assert codec_provider.codec_name == codec
    buffer_pool = TensorBufferPool(TensorBufferPoolConfig())

    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor = codec_provider.try_acquire_compressor()
    assert compressor is not None
    try:
        buffer = buffer_pool.try_acquire(compressor.compress_bound(source.numel()))
        assert buffer is not None
        compressed_numel = compressor.compress_into(source, buffer.tensor)
        assert compressed_numel < source.numel()
    finally:
        codec_provider.release(compressor)
    try:
        restored = torch.empty_like(source)
        decompressor = codec_provider.acquire_decompressor()
        try:
            decompressor.decompress_into(
                buffer.tensor[:compressed_numel], compressed_numel, restored
            )
        finally:
            codec_provider.release(decompressor)
        assert torch.equal(restored, source)
    finally:
        buffer.release()


def test_collective_group_prepares_compressed_cpu_tensors():
    """Prepared tensor data keeps raw entries and replaces compressed entries."""
    options = LZ4CompressionConfig()
    codec_provider = options.create_codec_provider()
    buffer_pool = TensorBufferPool(TensorBufferPoolConfig())
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_config=options,
        _tensor_buffer_pool=buffer_pool,
        _get_tensor_codec_provider=lambda: codec_provider,
    )
    fp32_tensor = torch.zeros(4096, dtype=torch.float32)
    uint8_tensor = torch.zeros(16 * 1024, dtype=torch.uint8)
    tensor_data = TensorData(
        cpu_tensor_mask=[True, True],
        cpu_tensors=[fp32_tensor, uint8_tensor],
        accel_tensors=[],
    )

    wire_data, buffers = group._compress_tensor_data(tensor_data)

    assert wire_data.compression is not None
    assert wire_data.compression.compressed_numel[0] is None
    assert wire_data.compression.compressed_numel[1] is not None
    assert wire_data.cpu_tensors[0] is fp32_tensor
    assert wire_data.cpu_tensors[1] is not uint8_tensor
    assert tensor_data.cpu_tensors[0] is fp32_tensor
    assert tensor_data.cpu_tensors[1] is uint8_tensor
    for buffer in buffers:
        buffer.release()


def test_collective_group_restores_a_compressed_cpu_tensor():
    """Tensor-list metadata restores a compressed CPU payload in place."""
    options = LZ4CompressionConfig(min_bytes=1)
    codec_provider = options.create_codec_provider()
    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor = codec_provider.try_acquire_compressor()
    assert compressor is not None
    capacity = compressor.compress_bound(source.numel())
    assert capacity is not None
    wire_tensor = torch.empty(capacity, dtype=torch.uint8)
    try:
        wire_numel = compressor.compress_into(source, wire_tensor)
    finally:
        codec_provider.release(compressor)

    metadata = {
        "meta": [(source.shape, source.dtype)],
        "pb": "payload",
        "cpu_tensor_mask": [True],
        "compression": TensorCompressionWireMetadata(
            codec=options.codec,
            compressed_numel=(wire_numel,),
        ),
    }
    incoming = iter(
        [
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, dtype=torch.uint8),
            wire_tensor[:wire_numel],
        ]
    )

    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_config=options,
        _tensor_buffer_pool=TensorBufferPool(TensorBufferPoolConfig()),
        _get_tensor_codec_provider=lambda: codec_provider,
    )
    group._peer_rank = 0
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)
    group._tensor_to_object = lambda *_args: metadata
    group._recv = lambda tensor, *_args, **_kwargs: tensor.copy_(next(incoming))

    tensors, piggyback_payload = group._recv_tensor_list(comm_id=0)

    assert piggyback_payload == "payload"
    assert torch.equal(tensors[0], source)


def test_float32_compression_can_be_explicitly_enabled():
    """An empty exclusion list restores dtype-agnostic compression."""
    options = LZ4CompressionConfig(min_bytes=1, excluded_dtypes=[])

    assert options.should_compress(torch.zeros(1, dtype=torch.float32))


def test_worker_lazily_shares_one_codec_provider():
    """Concurrent CollectiveGroups share one Worker-wide codec provider."""
    worker = object.__new__(Worker)
    worker._tensor_compression_config = LZ4CompressionConfig()
    worker._tensor_buffer_pool = TensorBufferPool(TensorBufferPoolConfig())
    worker._tensor_codec_provider = None
    worker._tensor_codec_provider_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as executor:
        codec_providers = list(
            executor.map(lambda _: worker._get_tensor_codec_provider(), range(16))
        )

    assert all(provider is codec_providers[0] for provider in codec_providers)
    assert worker._tensor_codec_provider is codec_providers[0]


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(torch.zeros(1, dtype=torch.uint8), id="below-min-bytes"),
        pytest.param(
            torch.zeros(16 * 1024 // 4, dtype=torch.float32),
            id="excluded-dtype",
        ),
    ],
)
def test_ineligible_tensors_do_not_initialize_the_codec_provider(tensor):
    """Raw CPU transfers do not require the configured codec library."""
    options = LZ4CompressionConfig()
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_config=options,
        _get_tensor_codec_provider=lambda: pytest.fail(
            "ineligible tensors initialized the codec provider"
        ),
    )
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )

    wire_data, buffers = group._compress_tensor_data(tensor_data)

    assert wire_data is tensor_data
    assert buffers == []


def test_lz4_compress_bound_returns_none_for_an_unsupported_input_size():
    """An input-size limit is a normal no-compression outcome."""
    codec = LZ4TensorCodec()

    assert codec.compress_bound(LZ4TensorCodec._MAX_INPUT_SIZE + 1) is None


def test_collective_group_options_exclude_tensor_compression():
    """Tensor compression is not a per-call collective option."""
    with pytest.raises(TypeError, match="tensor_compression"):
        CollectiveGroupOptions(tensor_compression=LZ4CompressionConfig())


def test_tensor_container_helpers_keep_async_send_without_unused_options():
    """Private send helpers retain their baseline async contract only."""
    send_helpers = [
        CollectiveGroup._send_tensor_list,
        CollectiveGroup._send_tensor_dict,
        CollectiveGroup._send_tensor_dataclass,
    ]
    recv_helpers = [
        CollectiveGroup._recv_tensor_list,
        CollectiveGroup._recv_tensor_dict,
        CollectiveGroup._recv_tensor_dataclass,
    ]

    for helper in send_helpers:
        parameters = inspect.signature(helper).parameters
        assert "async_op" in parameters
        assert "options" not in parameters
    for helper in recv_helpers:
        assert "options" not in inspect.signature(helper).parameters


def _cluster_config(collective):
    """Build a ClusterConfig from the public ``cluster`` YAML mapping."""
    return ClusterConfig.from_dict_cfg(
        OmegaConf.create(
            {"num_nodes": 1, "component_placement": [], "collective": collective}
        )
    )


@pytest.mark.parametrize(
    ("collective", "message"),
    [
        ({"tensor_compression": {"enabled": "yes"}}, "codec must be specified"),
        ({"tensor_compression": {"codec": "invalid"}}, "Unknown tensor compression"),
        ({"tensor_compression": {"codec": "lz4", "acceleration": 0}}, "acceleration"),
        ({"tensor_compression": {"codec": "zstd", "level": 0}}, "level"),
        ({"tensor_compression": {"codec": "lz4", "min_bytes": 0}}, "min_bytes"),
        ({"tensor_compression": {"codec": "zstd", "max_inflight": 0}}, "max_inflight"),
        (
            {"tensor_compression": {"codec": "lz4", "excluded_dtypes": "float32"}},
            "must be a list",
        ),
        (
            {"tensor_compression": {"codec": "lz4", "excluded_dtypes": ["nope"]}},
            "Unknown torch dtype",
        ),
        (
            {
                "tensor_compression": {
                    "codec": "lz4",
                    "excluded_dtypes": ["float32", "float32"],
                }
            },
            "duplicates",
        ),
        ({"tensor_compression": {"codec": "lz4", "min_bytes": 1.5}}, "integer"),
        ({"tensor_compression": {"codec": "lz4", "min_bytes": True}}, "integer"),
        ({"tensor_compression": {"codec": "lz4", "acceleration": 1.5}}, "integer"),
        ({"tensor_compression": {"codec": "zstd", "level": True}}, "integer"),
        ({"tensor_compression": {"codec": "zstd", "max_inflight": 4.0}}, "integer"),
        ({"tensor_buffer_pool": {"max_bytes": 0}}, "max_bytes must be >= 1"),
        ({"tensor_buffer_pool": {"max_bytes": 1.5}}, "integer"),
    ],
)
def test_cluster_config_validates_collective_settings(collective, message):
    """Invalid collective settings fail while the driver parses cluster yaml."""
    with pytest.raises(ValueError, match=message):
        _cluster_config(collective)


@pytest.mark.parametrize(
    ("collective", "message"),
    [
        ({"tensor_compresion": {"codec": "lz4"}}, "in cluster collective yaml config"),
        (
            {"tensor_compression": {"codec": "lz4", "min_byte": 1024}},
            "in cluster collective tensor_compression yaml config",
        ),
        (
            {"tensor_compression": {"codec": "lz4", "max_inflight": 4}},
            "in cluster collective tensor_compression yaml config",
        ),
        (
            {"tensor_compression": {"codec": "zstd", "acceleration": 1}},
            "in cluster collective tensor_compression yaml config",
        ),
        (
            {"tensor_buffer_pool": {"max_byte": 1024}},
            "in cluster collective tensor_buffer_pool yaml config",
        ),
    ],
)
def test_cluster_config_rejects_unknown_collective_keys(collective, message):
    """Unknown keys are reported the same way as any other cluster yaml typo."""
    with pytest.raises(AssertionError, match=message):
        _cluster_config(collective)


@pytest.mark.parametrize(
    ("collective", "message"),
    [
        ({"tensor_compression": True}, "tensor_compression must be a dictionary"),
        ({"tensor_buffer_pool": True}, "tensor_buffer_pool must be a dictionary"),
    ],
)
def test_cluster_config_requires_collective_mappings(collective, message):
    """Each collective sub-config must be a yaml mapping."""
    with pytest.raises(AssertionError, match=message):
        _cluster_config(collective)


def test_codec_configs_register_under_their_codec_type():
    """Every codec is discoverable by the name that selects it in yaml."""
    assert TensorCompressionManager.codec_config_register == {
        LZ4CompressionConfig.CODEC_TYPE: LZ4CompressionConfig,
        ZstdCompressionConfig.CODEC_TYPE: ZstdCompressionConfig,
    }
    assert LZ4CompressionConfig().create_codec_provider().codec_name == "lz4"
    assert ZstdCompressionConfig().create_codec_provider().codec_name == "zstd"


def test_registering_a_codec_config_requires_a_codec_type():
    """A config with no CODEC_TYPE could never be selected, so it cannot register."""
    with pytest.raises(AssertionError, match="CODEC_TYPE"):

        @TensorCompressionManager.register_codec_config
        @dataclass
        class UnnamedCompressionConfig(TensorCompressionConfig):
            """A codec config that forgot to name itself."""


def test_cluster_config_builds_the_selected_codec_config():
    """``codec`` selects the config class that owns the codec's parameters."""
    cluster_config = _cluster_config(
        {
            "tensor_buffer_pool": {"max_bytes": 4096},
            "tensor_compression": {
                "enabled": False,
                "codec": "zstd",
                "min_bytes": 1024,
                "excluded_dtypes": ["float32", "float64"],
                "level": 3,
                "max_inflight": 2,
            },
        }
    )

    assert cluster_config.collective == CollectiveConfig(
        tensor_compression=ZstdCompressionConfig(
            enabled=False,
            min_bytes=1024,
            excluded_dtypes=["float32", "float64"],
            level=3,
            max_inflight=2,
        ),
        tensor_buffer_pool=TensorBufferPoolConfig(max_bytes=4096),
    )
    assert cluster_config.collective.tensor_compression.codec == "zstd"


def test_cluster_config_supplies_the_default_tensor_buffer_pool():
    """Compression without an explicit pool block still gets the default budget."""
    cluster_config = _cluster_config({"tensor_compression": {"codec": "lz4"}})

    assert cluster_config.collective.tensor_buffer_pool == TensorBufferPoolConfig()
    assert cluster_config.collective.tensor_compression == LZ4CompressionConfig()


def test_cluster_config_without_collective_keeps_the_raw_wire_path():
    """Omitting ``cluster.collective`` leaves compression unconfigured."""
    cluster_config = ClusterConfig.from_dict_cfg(
        OmegaConf.create({"num_nodes": 1, "component_placement": []})
    )

    assert cluster_config.collective is None


def test_worker_loads_and_probes_collective_resources(monkeypatch):
    """Workers take their shared resources from the job-wide ClusterConfig."""
    worker = object.__new__(Worker)
    cluster_config = _cluster_config(
        {
            "tensor_buffer_pool": {"max_bytes": 4096},
            "tensor_compression": {"codec": "zstd", "min_bytes": 1024, "level": 3},
        }
    )
    monkeypatch.setattr(
        Cluster,
        "__new__",
        lambda _cls: SimpleNamespace(collective_config=cluster_config.collective),
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.scheduler.collective.tensor_compression.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert worker._tensor_compression_config == ZstdCompressionConfig(
        min_bytes=1024, level=3
    )
    assert worker._tensor_buffer_pool.config == TensorBufferPoolConfig(max_bytes=4096)
    assert probes == ["zstd"]
    assert worker._tensor_codec_provider is None


def test_worker_without_collective_config_uses_pool_defaults(monkeypatch):
    """A job that never configured collectives still gets a usable buffer pool."""
    worker = object.__new__(Worker)
    monkeypatch.setattr(
        Cluster, "__new__", lambda _cls: SimpleNamespace(collective_config=None)
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.scheduler.collective.tensor_compression.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert worker._tensor_compression_config is None
    assert worker._tensor_buffer_pool.config == TensorBufferPoolConfig()
    assert probes == []


def test_worker_skips_codec_resources_when_compression_is_disabled(monkeypatch):
    """Disabled compression neither probes nor creates codec resources."""
    worker = object.__new__(Worker)
    cluster_config = _cluster_config(
        {"tensor_compression": {"codec": "zstd", "enabled": False}}
    )
    monkeypatch.setattr(
        Cluster,
        "__new__",
        lambda _cls: SimpleNamespace(collective_config=cluster_config.collective),
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.scheduler.collective.tensor_compression.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert probes == []
    with pytest.raises(ValueError, match="not enabled"):
        worker._get_tensor_codec_provider()
    assert worker._tensor_codec_provider is None


def test_net_emulation_uses_the_compressed_wire_size():
    """Compression finishes before a point-to-point bandwidth reservation."""
    group = object.__new__(CollectiveGroup)
    tensor = torch.zeros(1024, dtype=torch.uint8)
    wire_tensor = torch.zeros(64, dtype=torch.uint8)
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )
    metadata = TensorCompressionWireMetadata(codec="lz4", compressed_numel=(64,))
    wire_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[wire_tensor],
        accel_tensors=[],
        compression=metadata,
    )
    events = []

    group._init_process_group = lambda **_kwargs: None
    group._compress_tensor_data = lambda _tensor_data: (
        events.append("compress") or wire_data,
        [],
    )
    group._wait_for_net_emulation = lambda *_payloads, size_bytes=None: events.append(
        ("reserve", size_bytes)
    )
    group._send = lambda *_args, **_kwargs: None
    group._send_tensor_list = lambda *_args, **_kwargs: events.append("send")
    group._cur_worker_address = SimpleNamespace(get_name=lambda: "Src:0")
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)
    group._net_emu_manager = object()

    group._atomic_send(
        work=None,
        object=tensor,
        comm_id=0,
        object_type=CollectiveGroup.TENSOR,
        tensor_data=tensor_data,
    )

    raw_size = group._estimate_payload_size((tensor, None))
    metadata_size = group._estimate_payload_size((metadata,))
    assert events == [
        "compress",
        ("reserve", raw_size - tensor.numel() + wire_tensor.numel() + metadata_size),
        "send",
    ]


def test_compressed_send_skips_size_estimation_without_net_emulation():
    """Disabled network emulation adds no payload-estimation overhead."""
    group = object.__new__(CollectiveGroup)
    tensor = torch.zeros(1024, dtype=torch.uint8)
    metadata = TensorCompressionWireMetadata(codec="lz4", compressed_numel=(64,))
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )
    wire_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[torch.zeros(64, dtype=torch.uint8)],
        accel_tensors=[],
        compression=metadata,
    )

    group._net_emu_manager = None
    group._init_process_group = lambda **_kwargs: None
    group._compress_tensor_data = lambda _tensor_data: (wire_data, [])
    group._estimate_payload_size = lambda *_args: pytest.fail(
        "payload size was estimated with network emulation disabled"
    )
    group._send = lambda *_args, **_kwargs: None
    group._send_tensor_list = lambda *_args, **_kwargs: None
    group._cur_worker_address = SimpleNamespace(get_name=lambda: "Src:0")
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)

    group._atomic_send(
        work=None,
        object=tensor,
        comm_id=0,
        object_type=CollectiveGroup.TENSOR,
        tensor_data=tensor_data,
    )
