"""grad-equivalent(GE) 비용 모델.

왜 wall-clock을 주 지표로 쓰지 않는가
-------------------------------------
MNIST MLP 784-128-10은 약 101,770 파라미터다. 이 규모에서 GPU 시간은 FLOP이
아니라 커널 런치와 파이썬 오버헤드에 지배된다. 이 상태에서 측정한 wall-clock은
최적화 효율이 아니라 구현 오버헤드를 반영한다. "RL이 wall-clock을 15% 줄였다"는
주장을 방어할 수 없다.

그래서 프로토콜 D1은 하드웨어 독립적인 비용 단위를 쓴다.

    1 GE = gradient batch 1회 forward + backward

Newton-CG step 하나의 비용은 다음으로 환산된다.

    cost_GE(k) = c_grad_graph + k * c_hvp + c_fwd

    c_grad_graph  create_graph=True 인 gradient 계산 (HVP를 위해 그래프 유지)
    c_hvp         그래프를 재사용하는 HVP 1회의 한계 비용
    c_fwd         step acceptance용 forward 1회
    k             CG 반복 수 = HVP 횟수

계수는 이론값(c_hvp ~ 2.5)을 쓰지 않고 **대상 하드웨어에서 실측**한다.

측정 원칙 (README §15)
----------------------
- warm-up 이후에 측정한다.
- 측정 구간 앞뒤로 ``torch.cuda.synchronize()`` 를 호출한다.
- peak VRAM은 ``torch.cuda.max_memory_allocated()`` 로 기록한다.
- 중앙값을 쓴다. 평균은 OS 스케줄링 스파이크에 취약하다.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

__all__ = ["CostModel", "measure_cost_model", "LAUNCH_BOUND_THRESHOLD_MS"]

HOST_ID_FALLBACK = "host-unspecified"
"""``EXPERIMENT_HOST_ID`` 가 없을 때 쓰는 라벨. 장치 실제 이름을 기록하지 않는다."""


LAUNCH_BOUND_THRESHOLD_MS = 1.0
"""gradient 1회가 이 시간보다 짧으면 런치 오버헤드 지배 구간으로 본다.

경험적 임계값이다. 이 구간에서는 wall-clock 기반 결론을 그대로 신뢰할 수 없고,
프로토콜 D1에 따라 GE 기준 결론을 주로 삼아야 한다.
"""


def _to_builtin(value: Any) -> Any:
    """``str`` / ``int`` / ``float`` 을 상속한 객체를 기본 타입으로 되돌린다.

    pyyaml SafeDumper 는 서브클래스를 표현하지 못한다. ``torch.__version__``
    (``TorchVersion``, ``str`` 서브클래스)이 대표적인 사례다.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(v) for v in value]
    return str(value)


@dataclass(slots=True)
class CostModel:
    """특정 (모델, 배치 구성, 하드웨어)에 대한 GE 환산 계수.

    Attributes:
        model_id: 이 계수가 유효한 구성의 식별자.
            예: ``"mnist_mlp_bg512_bc512_rtx3060ti"``.
        c_grad: 정의상 1.0. 기준 단위.
        c_grad_graph: ``create_graph=True`` gradient 비용 (GE).
        c_hvp: 그래프 재사용 HVP 1회 비용 (GE).
        c_fwd: forward 1회 비용 (GE).
        t_grad_ms: 실측 gradient 시간 중앙값 (ms). 진단용.
        t_grad_graph_ms: 실측 create_graph gradient 시간 중앙값 (ms).
        t_hvp_ms: 실측 HVP 시간 중앙값 (ms).
        t_fwd_ms: 실측 forward 시간 중앙값 (ms).
        n_params: 선형계 차원.
        grad_batch_size: gradient batch 크기.
        curvature_batch_size: curvature batch 크기.
        device: 측정 디바이스.
        gpu_name: GPU 이름.
        torch_version: torch 버전.
        n_repeat: 측정 반복 수.
        peak_vram_mb: 측정 중 peak VRAM.
    """

    model_id: str
    c_grad_graph: float
    c_hvp: float
    c_fwd: float
    t_grad_ms: float
    t_grad_graph_ms: float
    t_hvp_ms: float
    t_fwd_ms: float
    n_params: int
    grad_batch_size: int | None = None
    curvature_batch_size: int | None = None
    device: str = "cpu"
    gpu_name: str | None = None
    torch_version: str = ""
    host: str = ""
    n_repeat: int = 0
    peak_vram_mb: float | None = None
    c_grad: float = 1.0
    notes: dict[str, Any] = field(default_factory=dict)

    # --- 비용 환산 --------------------------------------------------------

    def newton_step_ge(self, cg_iters: int, *, include_acceptance: bool = True) -> float:
        """CG ``cg_iters`` 회를 쓴 Newton-CG step 1회의 GE 비용.

        Args:
            cg_iters: 이번 step에서 실제로 수행한 CG 반복 수 = HVP 횟수.
            include_acceptance: step acceptance forward를 포함할지.

        Returns:
            GE 단위 비용.

        Example:
            >>> cm = CostModel(model_id="t", c_grad_graph=1.4, c_hvp=2.0, c_fwd=0.3,
            ...                t_grad_ms=1.0, t_grad_graph_ms=1.4, t_hvp_ms=2.0,
            ...                t_fwd_ms=0.3, n_params=100)
            >>> cm.newton_step_ge(10)
            21.7
        """
        if cg_iters < 0:
            raise ValueError(f"cg_iters must be >= 0, got {cg_iters}")
        total = self.c_grad_graph + cg_iters * self.c_hvp
        if include_acceptance:
            total += self.c_fwd
        return total

    def first_order_step_ge(self) -> float:
        """AdamW / SGD 등 1차 optimizer step 1회의 GE 비용. 정의상 1.0."""
        return self.c_grad

    @property
    def is_launch_bound(self) -> bool:
        """런치 오버헤드 지배 구간인지 여부.

        ``True`` 이면 wall-clock 결론을 단독 근거로 쓰지 않는다 (프로토콜 D1).
        """
        return self.t_grad_ms < LAUNCH_BOUND_THRESHOLD_MS

    @property
    def hvp_flop_ratio_sanity(self) -> float:
        """``c_hvp`` 의 이론 기대치(약 2.0)에 대한 비율.

        1.0에서 크게 벗어나면 (특히 크게 작으면) 오버헤드 지배를 의심한다.
        double-backward는 forward+backward 대비 대략 2배 FLOP이 든다.
        """
        return self.c_hvp / 2.0

    # --- 직렬화 -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """YAML/JSON 로 안전하게 직렬화 가능한 dict 를 만든다.

        ``torch.__version__`` 처럼 ``str`` 을 상속한 객체(``TorchVersion``)는
        pyyaml SafeDumper 가 거부하므로 기본 타입으로 정규화한다.
        """
        d = _to_builtin(asdict(self))
        d["is_launch_bound"] = self.is_launch_bound
        return d

    def save(self, path: str | Path) -> Path:
        """YAML로 저장한다."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> CostModel:
        """YAML에서 읽어온다. ``is_launch_bound`` 같은 파생 필드는 무시한다."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        names = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})


# ---------------------------------------------------------------------------
# 측정
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_median_ms(
    fn: Callable[[], Any], device: torch.device, n_warmup: int, n_repeat: int
) -> float:
    """``fn`` 실행 시간의 중앙값(ms)을 측정한다.

    warm-up 후 측정하며, 각 반복의 앞뒤로 CUDA 동기화를 수행한다.
    """
    for _ in range(n_warmup):
        fn()
    _sync(device)

    samples: list[float] = []
    for _ in range(n_repeat):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def measure_cost_model(
    params: Sequence[Tensor],
    loss_fn_grad: Callable[[], Tensor],
    loss_fn_curv: Callable[[], Tensor] | None = None,
    *,
    model_id: str,
    grad_batch_size: int | None = None,
    curvature_batch_size: int | None = None,
    n_warmup: int = 5,
    n_repeat: int = 25,
) -> CostModel:
    """대상 하드웨어에서 GE 환산 계수를 실측한다.

    Args:
        params: 미분 대상 파라미터. ``requires_grad=True`` 여야 한다.
        loss_fn_grad: gradient batch에 대해 스칼라 loss를 반환하는 클로저.
            매 호출마다 새 그래프를 만들어야 한다.
        loss_fn_curv: curvature batch에 대한 loss 클로저. ``None`` 이면
            ``loss_fn_grad`` 를 쓴다 (``B_c = B_g``, 프로토콜 D2 기본값).
        model_id: 이 계수가 유효한 구성의 식별자.
        grad_batch_size: 기록용 메타데이터.
        curvature_batch_size: 기록용 메타데이터.
        n_warmup: 측정 전 warm-up 횟수.
        n_repeat: 측정 반복 횟수. 중앙값을 취한다.

    Returns:
        측정된 ``CostModel``.

    Raises:
        ValueError: ``params`` 가 비었거나 ``requires_grad=False`` 인 경우.
        RuntimeError: loss가 스칼라가 아니거나 그래프가 만들어지지 않은 경우.
    """
    plist = list(params)
    if not plist:
        raise ValueError("params must not be empty")
    if not all(p.requires_grad for p in plist):
        raise ValueError("all params must have requires_grad=True")

    device = plist[0].device
    curv_fn = loss_fn_curv if loss_fn_curv is not None else loss_fn_grad

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- 사전 검증: 클로저가 미분 가능한 스칼라를 주는지 확인 ---
    probe = loss_fn_grad()
    if probe.dim() != 0:
        raise RuntimeError(f"loss_fn_grad must return a scalar, got shape {tuple(probe.shape)}")
    if probe.grad_fn is None:
        raise RuntimeError(
            "loss_fn_grad returned a tensor with no grad_fn; "
            "the closure must build the graph on every call"
        )

    n_params = sum(p.numel() for p in plist)
    v = torch.randn(n_params, device=device, dtype=plist[0].dtype)

    # --- (1) forward only ---
    def _forward_only() -> None:
        with torch.no_grad():
            loss_fn_grad()

    # --- (2) gradient: 1 GE 의 정의 ---
    def _grad() -> None:
        loss = loss_fn_grad()
        torch.autograd.grad(loss, plist, allow_unused=True)

    # --- (3) create_graph=True gradient (HVP를 위해 그래프 유지) ---
    def _grad_with_graph() -> None:
        loss = curv_fn()
        torch.autograd.grad(loss, plist, create_graph=True, allow_unused=True)

    # --- (4) HVP 한계 비용: 그래프를 미리 만들어 두고 두 번째 backward만 측정 ---
    def _build_graph() -> Tensor:
        loss = curv_fn()
        grads = torch.autograd.grad(loss, plist, create_graph=True, allow_unused=True)
        pieces = [
            torch.zeros(p.numel(), device=device, dtype=p.dtype) if g is None else g.reshape(-1)
            for p, g in zip(plist, grads, strict=True)
        ]
        return torch.cat(pieces)

    gflat = _build_graph()

    def _hvp() -> None:
        torch.autograd.grad(gflat @ v, plist, retain_graph=True, allow_unused=True)

    t_fwd = _time_median_ms(_forward_only, device, n_warmup, n_repeat)
    t_grad = _time_median_ms(_grad, device, n_warmup, n_repeat)
    t_grad_graph = _time_median_ms(_grad_with_graph, device, n_warmup, n_repeat)
    t_hvp = _time_median_ms(_hvp, device, n_warmup, n_repeat)

    if t_grad <= 0.0:
        raise RuntimeError("measured gradient time is non-positive; timer resolution too coarse")

    peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    return CostModel(
        model_id=model_id,
        c_grad_graph=t_grad_graph / t_grad,
        c_hvp=t_hvp / t_grad,
        c_fwd=t_fwd / t_grad,
        t_grad_ms=t_grad,
        t_grad_graph_ms=t_grad_graph,
        t_hvp_ms=t_hvp,
        t_fwd_ms=t_fwd,
        n_params=n_params,
        grad_batch_size=grad_batch_size,
        curvature_batch_size=curvature_batch_size,
        device=str(device),
        gpu_name=gpu_name,
        torch_version=str(torch.__version__),
        # 장치의 실제 이름을 기록하지 않는다. `EXPERIMENT_HOST_ID` 별칭을 쓴다.
        # 재현에는 GPU 모델, torch 버전, batch 구성이 필요하고 장치 이름은 아니다.
        host=os.environ.get("EXPERIMENT_HOST_ID", HOST_ID_FALLBACK),
        n_repeat=n_repeat,
        peak_vram_mb=peak_vram,
    )
