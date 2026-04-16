from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

@dataclass(frozen=True)
class StarParams:
    M_star_sun: float = 1.0
    gamma_ms: float = 0.0

@dataclass(frozen=True)
class PlanetParams:
    P_days: float
    m_sin_i_mjup: float
    e: float
    inc_rad: float
    Omega_rad: float
    omega_rad: float
    l_rad: float
    m_true_mjup: Optional[float] = None

@dataclass(frozen=True)
class GPParams:
    use_gp: bool = False
    sigma_ms: float = 0.0
    period_days: float = 25.0
    Q0: float = 1.0
    dQ: float = 0.1
    f: float = 0.5

@dataclass(frozen=True)
class NoiseParams:
    sigma_white_ms: float = 1.0
    sigma_jitter_ms: float = 0.0
    gp: GPParams = GPParams()

@dataclass(frozen=True)
class InstrumentParams:
    label: str = "instA"
    gamma_ms: float = 0.0
    sigma_white_ms: float = 1.0
    sigma_jitter_ms: float = 0.0

@dataclass(frozen=True)
class ObservingSchedule:
    times_days: List[float]
    instruments: List[str]

@dataclass(frozen=True)
class SystemConfig:
    star: StarParams
    planets: List[PlanetParams]
    schedule: ObservingSchedule
    instruments: List[InstrumentParams] = field(default_factory=lambda: [InstrumentParams()])
    noise: NoiseParams = field(default_factory=NoiseParams)
    engine: str = "rebound"
    los_axis: str = "x"
    integrator_preference: str = "whfast"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Observations:
    times_days: List[float]
    rvs_ms: List[float]
    sigmas_ms: List[float]
    instruments: List[str]

@dataclass(frozen=True)
class Task:
    task_id: str
    config: SystemConfig
    observations: Observations
    truth_difficulty: int
    difficulty_details: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @staticmethod
    def from_json(s: str) -> "Task":
        d = json.loads(s)
        star = StarParams(**d["config"]["star"])
        planets = [PlanetParams(**p) for p in d["config"]["planets"]]
        schedule = ObservingSchedule(**d["config"]["schedule"])
        instruments = [InstrumentParams(**x) for x in d["config"].get("instruments", [])]
        noise_payload = d["config"].get("noise")
        if noise_payload is None:
            noise_payload = (d.get("meta") or {}).get("noise")
        if noise_payload is None:
            noise = NoiseParams()
        else:
            gp = GPParams(**(noise_payload.get("gp") or {}))
            noise = NoiseParams(
                sigma_white_ms=float(noise_payload.get("sigma_white_ms", 1.0)),
                sigma_jitter_ms=float(noise_payload.get("sigma_jitter_ms", 0.0)),
                gp=gp,
            )
        config = SystemConfig(
            star=star, planets=planets, schedule=schedule, instruments=instruments, noise=noise,
            engine=d["config"].get("engine","rebound"),
            los_axis=d["config"].get("los_axis","x"),
            integrator_preference=d["config"].get("integrator_preference","whfast"),
            metadata=d["config"].get("metadata",{}),
        )
        obs = Observations(**d["observations"])
        return Task(
            task_id=d["task_id"],
            config=config,
            observations=obs,
            truth_difficulty=d["truth_difficulty"],
            difficulty_details=d["difficulty_details"],
            meta=d.get("meta",{}),
        )
