"""Scripted baselines on the Python engine, mirroring game/src/agents (heuristic + random)."""
from __future__ import annotations

import numpy as np

from .engine import (LOOKAHEAD, MAX_SPEED, OBS_OFFSET, COIN_OFFSET, OBSTACLES_PER_LANE, LANE_COUNT, LANE_X,
                     LOW, HIGH, FULL, NOOP, JUMP, SLIDE, LEFT, RIGHT, Rng)


def _next_obstacle(s, lane, k=0):
    b = OBS_OFFSET + (lane * OBSTACLES_PER_LANE + k) * 4
    kind = LOW if s[b] else HIGH if s[b + 1] else FULL if s[b + 2] else -1
    return kind, float(s[b + 3]) * LOOKAHEAD


class HeuristicAgent:
    name = "heuristic"

    def act(self, s) -> int:
        lane = 0 if s[0] else 1 if s[1] else 2
        speed = float(s[8]) * MAX_SPEED
        airborne, sliding = s[4] > 0.5, s[6] > 0.5
        mid_tween = abs(float(s[3]) - LANE_X[lane]) > 0.05
        kind, dist = _next_obstacle(s, lane)
        if kind == FULL and dist < speed * 0.6 + 1:
            best = self._best_neighbour(s, lane, speed)
            if best != lane:
                return LEFT if best < lane else RIGHT
        if kind == LOW and not airborne and dist < speed * 0.2 + 0.4:
            return JUMP
        if kind == HIGH and not sliding and dist < speed * 0.25 + 0.4:
            return SLIDE
        if not airborne and not sliding and not mid_tween and dist > speed * 0.9:
            here = float(s[COIN_OFFSET + lane]) * LOOKAHEAD
            best, best_d = lane, here - 2
            for l in (lane - 1, lane + 1):
                if l < 0 or l >= LANE_COUNT:
                    continue
                d = float(s[COIN_OFFSET + l]) * LOOKAHEAD
                nk, nd = _next_obstacle(s, l)
                if (nk == -1 or nd > speed * 1.0) and d < best_d:
                    best, best_d = l, d
            if best != lane:
                return LEFT if best < lane else RIGHT
        return NOOP

    def _best_neighbour(self, s, lane, speed):
        best, best_score = lane, -np.inf
        for l in (lane - 1, lane + 1):
            if l < 0 or l >= LANE_COUNT:
                continue
            kind, dist = _next_obstacle(s, l)
            score = 1000 if kind == -1 else dist
            if kind in (LOW, HIGH):
                score += 50 if dist > speed * 0.15 else 0
            if score > best_score:
                best_score, best = score, l
        return best


class RandomAgent:
    name = "random"

    def __init__(self, seed=7, p_noop=0.85):
        self.rng, self.p_noop = Rng(seed), p_noop

    def act(self, s) -> int:
        if self.rng.next() < self.p_noop:
            return NOOP
        return 1 + self.rng.int(4)
