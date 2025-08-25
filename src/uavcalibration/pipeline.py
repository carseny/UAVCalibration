import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
import random
import time
from typing import Generic, Callable, Coroutine, TypeVar, Any, ParamSpec
from abc import abstractmethod

I = TypeVar("I")
O = TypeVar("O")
P = TypeVar("P")
R = TypeVar("R")
CTX = TypeVar("CTX")


class Worker(Generic[P, R, I, O]):
    def __init__(
        self,
        task: Callable[[P], R] | None = None,
        task_async: Callable[[P], Coroutine[Any, Any, R]] | None = None,
        preprocess: Callable[[I], P] | None = None,
        postprocess: Callable[[I, P, R], O] | None = None,
        input_queue: asyncio.Queue[I | None] | None = None,
        output_queue: asyncio.Queue[O | None] | None = None,
        executor: Executor | None = None,
    ) -> None:
        assert (task is None) ^ (task_async is None)
        self.task = task
        self.task_async = task_async
        self.preprocess = preprocess if preprocess is not None else lambda i: i
        self.postprocess = postprocess if postprocess is not None else lambda i, p, r: r
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.executor = executor

    async def mainloop(self):
        assert self.input_queue is not None
        loop = asyncio.get_running_loop()
        while True:
            input_ = await self.input_queue.get()
            # stop when input is None
            if input_ is None:
                if self.output_queue is not None:
                    # stop output queue
                    await self.output_queue.put(None)
                return
            # pre-process
            params = self.preprocess(input_)
            # run task
            if self.task_async is not None:
                processed = await self.task_async(params)
            elif self.task is not None:
                processed = await loop.run_in_executor(self.executor, self.task, params)
            else:
                raise ValueError("no task specificated")
            # post-process
            output = self.postprocess(input_, params, processed)
            if self.output_queue is not None:
                await self.output_queue.put(output)


class Stage(Generic[CTX, P, R]):
    def __init__(self, input_maxize: int = 10) -> None:
        super().__init__()
        self.input_queue = asyncio.Queue(input_maxize)
        self.output_queue: asyncio.Queue | None = None

    def connect_to(self, output_queue: asyncio.Queue):
        self.output_queue = output_queue

    def preprocess(self, ctx: CTX) -> P: ...
    def postprocess(self, ctx: CTX, p: P, r: R) -> CTX: ...
    def worker(self) -> Worker[P, R, CTX, CTX]: ...


class SyncStage(Stage[CTX, P, R]):
    @staticmethod
    @abstractmethod
    def task(args: P) -> R: ...

    def worker(self):
        return Worker(
            task=self.task,
            preprocess=self.preprocess,
            postprocess=self.postprocess,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )


class AsyncStage(Stage[CTX, P, R]):
    @staticmethod
    @abstractmethod
    async def task_async(args: P) -> R: ...

    def worker(self):
        return Worker(
            task_async=self.task_async,
            preprocess=self.preprocess,
            postprocess=self.postprocess,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )


class Pipeline:
    def __init__(
        self,
        workers: list[Worker],
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue | None = None,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self.workers = workers
        self._worker_tasks: list[asyncio.tasks.Task] = []
        self._running = False

    @classmethod
    def from_stages(cls, stages: list[Stage]) -> "Pipeline":
        for i, stage in enumerate(stages[:-1]):
            if stage.output_queue is None:
                stage.connect_to(stages[i + 1].input_queue)
        workers = [stage.worker() for stage in stages]
        return Pipeline(workers, stages[0].input_queue)

    @property
    def running(self):
        return self._running

    @property
    def input_queue(self):
        return self._output_queue

    @input_queue.setter
    def input_queue(self, value: asyncio.Queue):
        assert self._running == False, "Cannot set output_queue when running"
        self._output_queue = value
        self.workers[0].output_queue = value

    @property
    def output_queue(self):
        return self._output_queue

    @output_queue.setter
    def output_queue(self, value: asyncio.Queue | None):
        assert self._running == False, "Cannot set output_queue when running"
        self._output_queue = value
        self.workers[-1].output_queue = value

    async def start(self):
        """启动工作协程"""
        if not self._running:
            self._worker_tasks = [
                asyncio.create_task(worker.mainloop()) for worker in self.workers
            ]
            self._running = True

    async def stop(self):
        """停止流水线"""
        if self._running:
            await self.add_input(None)
            await asyncio.gather(*self._worker_tasks)
            self._worker_tasks.clear()
            self._running = False

    async def add_input(self, item):
        assert self._running
        await self._input_queue.put(item)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()


# # 示例任务函数
# def task_A(batch):
#     """CPU密集型任务示例"""
#     # 模拟CPU处理时间
#     time.sleep(0.05 * len(batch))
#     return [x * 2 for x in batch]


# async def task_B(batch):
#     """IO密集型任务示例"""
#     # 模拟IO等待时间
#     await asyncio.sleep(0.03 * len(batch))
#     return [x + 100 for x in batch]


# def task_C(batch):
#     """GPU密集型任务示例"""
#     # 模拟GPU处理时间
#     time.sleep(0.02 * len(batch))
#     return [x**2 for x in batch]


# class StreamingPipeline:
#     def __init__(self, batch_size=32, max_workers=4):
#         self.batch_size = batch_size
#         self.max_workers = max_workers

#         # 创建队列
#         self.input_queue = asyncio.Queue(maxsize=1000)
#         self.output_queue = asyncio.Queue(maxsize=1000)

#         # 进程池
#         self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

#         # 控制变量
#         self.running = False
#         self.processing_tasks = 0

#     async def _producer(self):
#         """从输入队列收集批次"""
#         while self.running or self.processing_tasks > 0 or not self.input_queue.empty():
#             batch = []
#             # 等待第一个元素
#             if self.input_queue.empty() and self.running:
#                 try:
#                     item = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
#                     batch.append(item)
#                     self.processing_tasks += 1
#                 except asyncio.TimeoutError:
#                     continue

#             # 收集一批数据
#             while len(batch) < self.batch_size and not self.input_queue.empty():
#                 try:
#                     item = self.input_queue.get_nowait()
#                     batch.append(item)
#                     self.processing_tasks += 1
#                 except asyncio.QueueEmpty:
#                     break

#             if batch:
#                 await self.queue_a.put(batch)

#         # 发送结束信号
#         await self.queue_a.put(None)

#     async def _worker_A(self):
#         """处理CPU密集型任务"""
#         while True:
#             batch = await self.queue_a.get()
#             if batch is None:
#                 await self.queue_b.put(None)
#                 return

#             # 非阻塞调用进程池
#             processed = await asyncio.get_running_loop().run_in_executor(
#                 self.process_pool, task_A, batch
#             )
#             await self.queue_b.put(processed)

#     async def _worker_B(self):
#         """处理IO密集型任务"""
#         while True:
#             batch = await self.queue_b.get()
#             if batch is None:
#                 await self.queue_c.put(None)
#                 return

#             processed = await task_B(batch)
#             await self.queue_c.put(processed)

#     async def _worker_C(self):
#         """处理GPU密集型任务"""
#         while True:
#             batch = await self.queue_c.get()
#             if batch is None:
#                 return

#             # GPU任务使用独立进程
#             processed = await asyncio.get_running_loop().run_in_executor(
#                 self.process_pool, task_C, batch
#             )

#             # 将结果放入输出队列
#             for result in processed:
#                 await self.output_queue.put(result)

#             self.processing_tasks -= len(processed)

#     async def start(self):
#         """启动流水线"""
#         if self.running:
#             return

#         self.running = True
#         self.processing_tasks = 0

#         # 创建内部队列
#         self.queue_a = asyncio.Queue(maxsize=100)
#         self.queue_b = asyncio.Queue(maxsize=100)
#         self.queue_c = asyncio.Queue(maxsize=100)

#         # 启动工作协程
#         self.worker_tasks = [
#             asyncio.create_task(self._producer()),
#             asyncio.create_task(self._worker_A()),
#             asyncio.create_task(self._worker_B()),
#             asyncio.create_task(self._worker_C()),
#         ]

#     async def stop(self):
#         """停止流水线"""
#         if not self.running:
#             return

#         self.running = False
#         await asyncio.gather(*self.worker_tasks)
#         self.process_pool.shutdown(wait=True)

#     async def add_input(self, item):
#         """添加输入项"""
#         if not self.running:
#             raise RuntimeError("Pipeline is not running")
#         await self.input_queue.put(item)

#     async def get_output(self):
#         """获取输出项"""
#         return await self.output_queue.get()

#     def get_output_nowait(self):
#         """非阻塞获取输出项"""
#         return self.output_queue.get_nowait()


# # 测试函数
# async def test_pipeline():
#     pipeline = StreamingPipeline(batch_size=5, max_workers=4)
#     await pipeline.start()

#     # 创建任务收集输出
#     async def output_collector():
#         results = []
#         while True:
#             try:
#                 result = pipeline.get_output_nowait()
#                 results.append(result)
#                 print(f"Output received: {result}")
#             except asyncio.QueueEmpty:
#                 if not pipeline.running and pipeline.processing_tasks == 0:
#                     break
#                 await asyncio.sleep(0.1)
#         return results

#     collector_task = asyncio.create_task(output_collector())

#     # 模拟实时输入
#     for i in range(20):
#         await pipeline.add_input(i)
#         print(f"Input added: {i}")
#         await asyncio.sleep(random.uniform(0.05, 0.2))  # 随机间隔添加

#     # 等待所有输入处理完成
#     while pipeline.processing_tasks > 0:
#         await asyncio.sleep(0.1)

#     # 停止流水线并收集结果
#     await pipeline.stop()
#     results = await collector_task

#     print("\nFinal results:")
#     print(results)


# if __name__ == "__main__":
#     asyncio.run(test_pipeline())
