import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import time


# 示例任务函数
def task_A(batch):
    """CPU密集型任务示例"""
    # 模拟CPU处理时间
    time.sleep(0.05 * len(batch))
    return [x * 2 for x in batch]


async def task_B(batch):
    """IO密集型任务示例"""
    # 模拟IO等待时间
    await asyncio.sleep(0.03 * len(batch))
    return [x + 100 for x in batch]


def task_C(batch):
    """GPU密集型任务示例"""
    # 模拟GPU处理时间
    time.sleep(0.02 * len(batch))
    return [x**2 for x in batch]


class StreamingPipeline:
    def __init__(self, batch_size=32, max_workers=4):
        self.batch_size = batch_size
        self.max_workers = max_workers

        # 创建队列
        self.input_queue = asyncio.Queue(maxsize=1000)
        self.output_queue = asyncio.Queue(maxsize=1000)

        # 进程池
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)

        # 控制变量
        self.running = False
        self.processing_tasks = 0

    async def _producer(self):
        """从输入队列收集批次"""
        while self.running or self.processing_tasks > 0 or not self.input_queue.empty():
            batch = []
            # 等待第一个元素
            if self.input_queue.empty() and self.running:
                try:
                    item = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                    batch.append(item)
                    self.processing_tasks += 1
                except asyncio.TimeoutError:
                    continue

            # 收集一批数据
            while len(batch) < self.batch_size and not self.input_queue.empty():
                try:
                    item = self.input_queue.get_nowait()
                    batch.append(item)
                    self.processing_tasks += 1
                except asyncio.QueueEmpty:
                    break

            if batch:
                await self.queue_a.put(batch)

        # 发送结束信号
        await self.queue_a.put(None)

    async def _worker_A(self):
        """处理CPU密集型任务"""
        while True:
            batch = await self.queue_a.get()
            if batch is None:
                await self.queue_b.put(None)
                return

            # 非阻塞调用进程池
            processed = await asyncio.get_running_loop().run_in_executor(
                self.process_pool, task_A, batch
            )
            await self.queue_b.put(processed)

    async def _worker_B(self):
        """处理IO密集型任务"""
        while True:
            batch = await self.queue_b.get()
            if batch is None:
                await self.queue_c.put(None)
                return

            processed = await task_B(batch)
            await self.queue_c.put(processed)

    async def _worker_C(self):
        """处理GPU密集型任务"""
        while True:
            batch = await self.queue_c.get()
            if batch is None:
                return

            # GPU任务使用独立进程
            processed = await asyncio.get_running_loop().run_in_executor(
                self.process_pool, task_C, batch
            )

            # 将结果放入输出队列
            for result in processed:
                await self.output_queue.put(result)

            self.processing_tasks -= len(processed)

    async def start(self):
        """启动流水线"""
        if self.running:
            return

        self.running = True
        self.processing_tasks = 0

        # 创建内部队列
        self.queue_a = asyncio.Queue(maxsize=100)
        self.queue_b = asyncio.Queue(maxsize=100)
        self.queue_c = asyncio.Queue(maxsize=100)

        # 启动工作协程
        self.worker_tasks = [
            asyncio.create_task(self._producer()),
            asyncio.create_task(self._worker_A()),
            asyncio.create_task(self._worker_B()),
            asyncio.create_task(self._worker_C()),
        ]

    async def stop(self):
        """停止流水线"""
        if not self.running:
            return

        self.running = False
        await asyncio.gather(*self.worker_tasks)
        self.process_pool.shutdown(wait=True)

    async def add_input(self, item):
        """添加输入项"""
        if not self.running:
            raise RuntimeError("Pipeline is not running")
        await self.input_queue.put(item)

    async def get_output(self):
        """获取输出项"""
        return await self.output_queue.get()

    def get_output_nowait(self):
        """非阻塞获取输出项"""
        return self.output_queue.get_nowait()


# 测试函数
async def test_pipeline():
    pipeline = StreamingPipeline(batch_size=5, max_workers=4)
    await pipeline.start()

    # 创建任务收集输出
    async def output_collector():
        results = []
        while True:
            try:
                result = pipeline.get_output_nowait()
                results.append(result)
                print(f"Output received: {result}")
            except asyncio.QueueEmpty:
                if not pipeline.running and pipeline.processing_tasks == 0:
                    break
                await asyncio.sleep(0.1)
        return results

    collector_task = asyncio.create_task(output_collector())

    # 模拟实时输入
    for i in range(20):
        await pipeline.add_input(i)
        print(f"Input added: {i}")
        await asyncio.sleep(random.uniform(0.05, 0.2))  # 随机间隔添加

    # 等待所有输入处理完成
    while pipeline.processing_tasks > 0:
        await asyncio.sleep(0.1)

    # 停止流水线并收集结果
    await pipeline.stop()
    results = await collector_task

    print("\nFinal results:")
    print(results)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
