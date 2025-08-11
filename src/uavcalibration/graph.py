from abc import ABC, abstractmethod
from typing import TypeVar, ParamSpec, Any, Generic, Callable
import itertools

__all__ = ["ComputeNode", "FunctionNode", "InputNode"]

P = ParamSpec("P")
R = TypeVar("R")


class ComputeNode(ABC, Generic[P, R]):
    def __init__(
        self,
        *predecessors: "ComputeNode[..., Any]",
        use_cache=True,
        **kwpredecessors: "ComputeNode[..., Any]",
    ):
        # track successors
        self.successors: list[ComputeNode[..., Any]] = []
        self.predecessors = predecessors
        self.kwpredecessors = kwpredecessors
        # update predecessors
        for pred in itertools.chain(self.predecessors, self.kwpredecessors.values()):
            pred.successors.append(self)
        # whether use cache value
        self.use_cache = use_cache
        # cache value
        self._value: R | None = None

    @abstractmethod
    def compute(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    @property
    def value(self):
        # if cached
        if self._value is not None:
            return self._value
        # compute inputs args
        args = tuple(node.value for node in self.predecessors)
        kwargs = {k: v.value for k, v in self.kwpredecessors.items()}
        result = self.compute(*args, **kwargs)
        # update cache
        if self.use_cache:
            self._value = result
        return result

    @value.setter
    def value(self, value: R | None):
        if value != self._value:
            # clear cache
            for succ in self.successors:
                succ.value = None
        self._value = value


class FunctionNode(ComputeNode[P, R]):
    def __init__(self, func: Callable[P, R] | None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def compute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        assert self.func is not None, "Function has not set yet!"
        return self.func(*args, **kwargs)


class InputNode(ComputeNode[..., R]):
    def __init__(
        self,
        default: R | None = None,
        default_factory: Callable[..., R] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.default = default
        self.default_factory = default_factory

    def compute(self) -> R:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is not None:
            return self.default
        raise RuntimeError("No value or default value given!")
