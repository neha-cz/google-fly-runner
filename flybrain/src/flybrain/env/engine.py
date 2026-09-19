"""Python port of the FlyRunner engine (game/src/engine). The TypeScript engine is the source of
truth; this file mirrors it statement for statement, including PRNG call order, and is checked
against game/fixtures/golden.json by tests/test_engine.py. Change both or neither.
"""
from __future__ import annotations

import math

DT = 1 / 60
LANE_COUNT = 3
LANE_X = (-1.0, 0.0, 1.0)
BASE_SPEED, MAX_SPEED, SPEED_PER_METER = 8.0, 22.0, 0.015
JUMP_DURATION, JUMP_HEIGHT, LOW_HEIGHT = 0.55, 1.0, 0.3
SLIDE_DURATION, LANE_CHANGE_DURATION = 0.6, 0.2
PLAYER_HALF_DEPTH, PLAYER_HALF_WIDTH, OBSTACLE_HALF_WIDTH = 0.25, 0.3, 0.45
OBSTACLE_DEPTH = (0.6, 0.6, 0.8)
COIN_RADIUS, COIN_VALUE = 0.35, 10
SPAWN_AHEAD, DESPAWN_BEHIND, LOOKAHEAD = 70.0, 3.0, 40.0
LOW, HIGH, FULL = 0, 1, 2
ACTIONS = ("noop", "jump", "slide", "left", "right")
NOOP, JUMP, SLIDE, LEFT, RIGHT = range(5)
OBSTACLES_PER_LANE = 2
STATE_SIZE = 9 + LANE_COUNT * OBSTACLES_PER_LANE * 4 + LANE_COUNT
OBS_OFFSET, COIN_OFFSET = 9, 9 + LANE_COUNT * OBSTACLES_PER_LANE * 4
_M32 = 0xFFFFFFFF


def speed_at(distance: float) -> float:
    return min(MAX_SPEED, BASE_SPEED + distance * SPEED_PER_METER)


def _smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


class Rng:
    """xorshift32, identical to game/src/engine/rng.ts."""
    __slots__ = ("s",)

    def __init__(self, seed: int):
        self.s = (seed & _M32) or 0x9E3779B9

    def next_u32(self) -> int:
        x = self.s
        x ^= (x << 13) & _M32
        x ^= x >> 17
        x ^= (x << 5) & _M32
        self.s = x
        return x

    def next(self) -> float:
        return self.next_u32() / 4294967296

    def range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next()

    def int(self, n: int) -> int:
        return self.next_u32() % n


class Obstacle:
    __slots__ = ("kind", "lane", "z", "depth")

    def __init__(self, kind: int, lane: int, z: float, depth: float):
        self.kind, self.lane, self.z, self.depth = kind, lane, z, depth


class Coin:
    __slots__ = ("lane", "z", "taken")

    def __init__(self, lane: int, z: float):
        self.lane, self.z, self.taken = lane, z, False


class Spawner:
    __slots__ = ("rng", "next_wave_z", "first_wave")

    def __init__(self, rng: Rng, start_z: float):
        self.rng = rng
        self.next_wave_z = start_z + 18
        self.first_wave = True

    def update(self, player_z: float, distance: float, obstacles: list, coins: list) -> None:
        while self.next_wave_z < player_z + SPAWN_AHEAD:
            d_at_wave = distance + (self.next_wave_z - player_z)
            v = speed_at(d_at_wave)
            wave_depth = self._spawn_wave(self.next_wave_z, d_at_wave, obstacles)
            difficulty = max(0.5, 1 - d_at_wave / 1500)
            gap = max(v * 0.55, v * self.rng.range(0.9, 1.5) * difficulty)
            self._maybe_coin_trail(self.next_wave_z + wave_depth, gap - wave_depth, coins)
            self.next_wave_z += gap

    def _spawn_wave(self, z: float, distance: float, obstacles: list) -> float:
        rng = self.rng
        r = rng.next()
        p3 = 0 if self.first_wave else min(0.2, distance / 3000)
        p2 = 0 if self.first_wave else min(0.6, 0.15 + distance / 1000)
        n_lanes = 3 if r < p3 else 2 if r < p3 + p2 else 1
        self.first_wave = False
        if n_lanes == 3:
            lanes = [0, 1, 2]
        elif n_lanes == 2:
            free = rng.int(LANE_COUNT)
            lanes = [l for l in range(LANE_COUNT) if l != free]
        else:
            lanes = [rng.int(LANE_COUNT)]
        max_depth = 0.0
        for lane in lanes:
            k = rng.next()
            kind = LOW if k < 0.4 else HIGH if k < 0.7 else FULL
            if n_lanes == 3 and kind == FULL:
                kind = LOW if k < 0.85 else HIGH
            depth = OBSTACLE_DEPTH[kind]
            max_depth = max(max_depth, depth)
            obstacles.append(Obstacle(kind, lane, z, depth))
        return max_depth

    def _maybe_coin_trail(self, start_z: float, room: float, coins: list) -> None:
        rng = self.rng
        if rng.next() > 0.6:
            return
        spacing = 0.9
        n = min(3 + rng.int(4), math.floor((room - 2.0) / spacing))
        if n < 2:
            return
        lane = rng.int(LANE_COUNT)
        for i in range(n):
            coins.append(Coin(lane, start_z + 1.0 + i * spacing))


class Runner:
    """One game instance. Same API as the TS Runner: act/step/get_state/reset."""
    __slots__ = ("rng", "spawner", "obstacles", "coins", "time", "alive", "distance", "score", "coins_collected", "speed",
                 "crashed_into", "px", "py", "pz", "lane", "lane_t", "from_x", "lean_dir", "jump_t", "slide_t", "pending")

    def __init__(self, seed: int = 1):
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = Rng(seed)
        self.obstacles, self.coins = [], []
        self.time = 0.0; self.alive = True; self.distance = 0.0; self.score = 0.0; self.coins_collected = 0
        self.speed = speed_at(0); self.crashed_into = None
        self.px = 0.0; self.py = 0.0; self.pz = 0.0; self.lane = 1; self.lane_t = 1.0; self.from_x = 0.0; self.lean_dir = 0
        self.jump_t = -1.0; self.slide_t = -1.0; self.pending = NOOP
        self.spawner = Spawner(self.rng, 0)
        self.spawner.update(self.pz, self.distance, self.obstacles, self.coins)

    @property
    def airborne(self) -> bool:
        return self.jump_t >= 0

    @property
    def sliding(self) -> bool:
        return self.slide_t >= 0

    def act(self, a: int) -> None:
        self.pending = a

    def _apply(self, a: int) -> None:
        if a == JUMP:
            if not self.airborne:
                self.slide_t = -1.0; self.jump_t = 0.0
        elif a == SLIDE:
            if self.airborne:
                self.jump_t = -1.0; self.py = 0.0
            if not self.sliding:
                self.slide_t = 0.0
        elif a == LEFT:
            self._change_lane(-1)
        elif a == RIGHT:
            self._change_lane(1)

    def _change_lane(self, d: int) -> None:
        target = self.lane + d
        if target < 0 or target >= LANE_COUNT:
            return
        self.from_x = self.px; self.lane = target; self.lane_t = 0.0; self.lean_dir = d

    def step(self) -> tuple[bool, bool, int, float]:
        """Advance one tick. Returns (alive, crashed, coins_this_step, dz)."""
        if not self.alive:
            return False, False, 0, 0.0
        a = self.pending; self.pending = NOOP
        self._apply(a)
        self.time += DT
        self.speed = speed_at(self.distance)
        dz = self.speed * DT
        self.distance += dz; self.pz += dz; self.score += dz
        if self.lane_t < 1:
            self.lane_t = min(1.0, self.lane_t + DT / LANE_CHANGE_DURATION)
            self.px = self.from_x + (LANE_X[self.lane] - self.from_x) * _smoothstep(self.lane_t)
            if self.lane_t >= 1:
                self.px = LANE_X[self.lane]; self.lean_dir = 0
        if self.jump_t >= 0:
            self.jump_t += DT
            if self.jump_t >= JUMP_DURATION:
                self.jump_t = -1.0; self.py = 0.0
            else:
                self.py = JUMP_HEIGHT * math.sin(math.pi * self.jump_t / JUMP_DURATION)
        if self.slide_t >= 0:
            self.slide_t += DT
            if self.slide_t >= SLIDE_DURATION:
                self.slide_t = -1.0
        self.spawner.update(self.pz, self.distance, self.obstacles, self.coins)

        px, py, pz = self.px, self.py, self.pz
        z_lo, z_hi = pz - PLAYER_HALF_DEPTH, pz + PLAYER_HALF_DEPTH
        crashed = False
        sliding = self.slide_t >= 0
        for o in self.obstacles:
            if o.z > z_hi or o.z + o.depth < z_lo:
                continue
            if abs(px - LANE_X[o.lane]) >= PLAYER_HALF_WIDTH + OBSTACLE_HALF_WIDTH:
                continue
            if o.kind == LOW and py > LOW_HEIGHT:
                continue
            if o.kind == HIGH and sliding:
                continue
            crashed = True; self.crashed_into = o
            break
        got = 0
        for c in self.coins:
            if c.taken or abs(c.z - pz) > COIN_RADIUS + PLAYER_HALF_DEPTH:
                continue
            if abs(px - LANE_X[c.lane]) >= PLAYER_HALF_WIDTH + COIN_RADIUS:
                continue
            c.taken = True; got += 1
        self.coins_collected += got; self.score += got * COIN_VALUE

        cut = pz - DESPAWN_BEHIND
        obs = self.obstacles
        if obs and obs[0].z + obs[0].depth < cut:
            i = 0
            while i < len(obs) and obs[i].z + obs[i].depth < cut:
                i += 1
            del obs[:i]
        cs = self.coins
        if cs and cs[0].z < cut:
            i = 0
            while i < len(cs) and cs[i].z < cut:
                i += 1
            del cs[:i]
        if crashed:
            self.alive = False
        return self.alive, crashed, got, dz

    def get_state(self, out) -> None:
        """Fill a float32 array of STATE_SIZE exactly like state.ts."""
        out[:] = 0
        out[self.lane] = 1
        out[3] = self.px / LANE_X[LANE_COUNT - 1]
        out[4] = 1 if self.jump_t >= 0 else 0
        out[5] = self.jump_t / JUMP_DURATION if self.jump_t >= 0 else 0
        out[6] = 1 if self.slide_t >= 0 else 0
        out[7] = self.slide_t / SLIDE_DURATION if self.slide_t >= 0 else 0
        out[8] = self.speed / MAX_SPEED
        for l in range(LANE_COUNT):
            for k in range(OBSTACLES_PER_LANE):
                out[OBS_OFFSET + (l * OBSTACLES_PER_LANE + k) * 4 + 3] = 1
        count = [0, 0, 0]
        pz = self.pz
        for o in self.obstacles:
            l = o.lane
            if count[l] >= OBSTACLES_PER_LANE:
                continue
            if o.z + o.depth < pz:
                continue
            d = max(0.0, o.z - pz)
            if d > LOOKAHEAD:
                continue
            base = OBS_OFFSET + (l * OBSTACLES_PER_LANE + count[l]) * 4
            out[base + o.kind] = 1
            out[base + 3] = d / LOOKAHEAD
            count[l] += 1
        for l in range(LANE_COUNT):
            out[COIN_OFFSET + l] = 1
        seen = [False, False, False]
        for c in self.coins:
            if c.taken or seen[c.lane] or c.z < pz:
                continue
            d = c.z - pz
            if d > LOOKAHEAD:
                continue
            out[COIN_OFFSET + c.lane] = d / LOOKAHEAD; seen[c.lane] = True
