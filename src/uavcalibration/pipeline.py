import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Generic, Callable, Coroutine, TypeVar, Any, cast
from abc import abstractmethod

__all__ = ["Worker", "Pipeline", "Stage", "SyncStage", "AsyncStage"]

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type
P = TypeVar("P")  # Parameter Type
R = TypeVar("R")  # Return Type
CTX = TypeVar("CTX")


class Worker(Generic[I, O, P, R]):
    def __init__(
        self,
        input_queue: asyncio.Queue[I | None],
        output_queue: asyncio.Queue[O | None] | None = None,
        task: Callable[[P], R] | None = None,
        task_async: Callable[[P], Coroutine[Any, Any, R]] | None = None,
        preprocess: Callable[[I], P] | None = None,
        postprocess: Callable[[I, P, R], O] | None = None,
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


class Stage(Generic[I, O, P, R]):
    def __init__(self, input_maxsize: int = 10) -> None:
        super().__init__()
        self.input_queue = asyncio.Queue(input_maxsize)
        self.output_queue: asyncio.Queue | None = None

    def connect_to(self, output_queue: asyncio.Queue):
        self.output_queue = output_queue

    @abstractmethod
    def worker(self) -> Worker[I, O, P, R]: ...

    def preprocess(self, i: I) -> P:
        return cast(P, i)

    def postprocess(self, i: I, p: P, r: R) -> O:
        return cast(O, r)


class SyncStage(Stage[I, O, P, R]):
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


class AsyncStage(Stage[I, O, P, R]):
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


class Pipeline(Generic[I, O]):
    def __init__(
        self,
        workers: list[
            Worker[I | None, Any, Any, Any]
            | Worker[Any, O | None, Any, Any]
            | Worker[Any, Any, Any, Any]
        ],
    ) -> None:
        self.workers = workers
        self._worker_tasks: list[asyncio.tasks.Task] = []
        self._running = False

    @classmethod
    def from_stages(cls, stages: list[Stage]) -> "Pipeline":
        for i, stage in enumerate(stages[:-1]):
            if stage.output_queue is None:
                stage.connect_to(stages[i + 1].input_queue)
        workers = [stage.worker() for stage in stages]
        return Pipeline(workers)

    def prepend(self, worker: Worker):
        assert self._running == False, "Please stop pipeline before modifying it"
        worker.output_queue = self.workers[0].input_queue
        self.workers.insert(0, worker)

    def append(self, worker: Worker):
        assert self._running == False, "Please stop pipeline before modifying it"
        self.workers[-1].output_queue = worker.input_queue
        self.workers.append(worker)

    @property
    def running(self):
        return self._running

    @property
    def input_queue(self) -> asyncio.Queue[I | None]:
        input_queue = self.workers[0].input_queue
        return cast(asyncio.Queue[I | None], input_queue)

    @input_queue.setter
    def input_queue(self, value: asyncio.Queue[I | None]):
        assert self._running == False, "Please stop pipeline before modifying it"
        self.workers[0].input_queue = value

    @property
    def output_queue(self) -> asyncio.Queue[O | None] | None:
        output_queue = self.workers[-1].output_queue
        return cast(asyncio.Queue[O | None] | None, output_queue)

    @output_queue.setter
    def output_queue(self, value: asyncio.Queue[O | None] | None):
        assert self._running == False, "Please stop pipeline before modifying it"
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
        await self.input_queue.put(item)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
