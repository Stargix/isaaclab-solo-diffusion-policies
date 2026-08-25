import argparse
from pathlib import Path

import numpy as np


WALK_HEIGHT = 0.2932
CROUCH_HEIGHT = 0.1705


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3D paths for Solo12 Diffusion Policy evaluation.")
    parser.add_argument(
        "--shape", type=str, default="straight",
        choices=["straight", "s_curve", "circle", "right_angle", "mixed", "sharp_v_curves"],
        help="Shape of the path.",
    )
    parser.add_argument("--length", type=float, default=8.0, help="Total length of the path in meters.")
    parser.add_argument("--step_size", type=float, default=0.05, help="Step size between path points.")
    parser.add_argument("--z_start", type=float, default=WALK_HEIGHT, help="Initial walk height.")
    parser.add_argument("--z_end", type=float, default=CROUCH_HEIGHT, help="Target crouch height.")
    parser.add_argument("--transition_x", type=float, default=3.0, help="X position where height transitions from z_start to z_end.")
    parser.add_argument(
        "--height_cycle", type=float, nargs="+", default=None,
        help="Optional repeated height schedule, e.g. 0.2932 0.25 0.21 0.1705.",
    )
    parser.add_argument(
        "--height_segment_m", type=float, default=1.0,
        help="Arc-length of each segment when --height_cycle is used.",
    )
    parser.add_argument("--output", type=str, default="scripts/diffusion_policy/paths/test_path.npy", help="Output file path (.npy).")
    args = parser.parse_args()

    if args.step_size <= 0.0 or args.length <= 0.0:
        raise ValueError("--length and --step_size must be positive.")
    if args.height_segment_m <= 0.0:
        raise ValueError("--height_segment_m must be positive.")

    steps = max(3, int(args.length / args.step_size) + 1)
    distance = np.linspace(0.0, args.length, steps, dtype=np.float32)
    if args.shape == "s_curve":
        x = distance
        y = 0.2 * np.sin(2.0 * np.pi * distance / args.length)
    elif args.shape == "circle":
        radius = args.length / (2.0 * np.pi)
        theta = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32)
        x = radius * np.sin(theta)
        y = radius * (1.0 - np.cos(theta))
    elif args.shape == "right_angle":
        half = max(2, steps // 2)
        x = np.concatenate((
            np.linspace(0.0, args.length / 2.0, half),
            np.full(steps - half, args.length / 2.0),
        ))
        y = np.concatenate((
            np.zeros(half),
            np.linspace(0.0, args.length / 2.0, steps - half),
        ))
    elif args.shape == "mixed":
        # Compact geometry diagnostic: straight -> 90-degree arc -> straight
        # -> second 90-degree arc -> smooth lateral curve.  It is generated
        # locally (first point [0, 0]) and rescaled to --length below.
        radius = 0.45
        samples_per_segment = 80

        def line(start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
            return np.stack(
                (
                    np.linspace(start[0], end[0], samples_per_segment),
                    np.linspace(start[1], end[1], samples_per_segment),
                ),
                axis=1,
            )

        u = np.linspace(0.0, np.pi / 2.0, samples_per_segment)
        first_turn = np.stack(
            (0.9 + radius * np.sin(u), radius - radius * np.cos(u)), axis=1
        )
        second_turn = np.stack(
            (0.9 + radius * np.cos(u), 1.25 + radius * np.sin(u)), axis=1
        )
        curve_u = np.linspace(0.0, 1.0, samples_per_segment)
        smooth_curve = np.stack(
            (
                0.9 - 1.4 * curve_u,
                1.25 + radius + 0.22 * (1.0 - np.cos(2.0 * np.pi * curve_u)) / 2.0,
            ),
            axis=1,
        )
        raw_xy = np.concatenate(
            (
                line((0.0, 0.0), (0.9, 0.0)),
                first_turn[1:],
                line((1.35, 0.45), (1.35, 1.25))[1:],
                second_turn[1:],
                smooth_curve[1:],
            ),
            axis=0,
        )
        raw_arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(raw_xy, axis=0), axis=1)))
        )
        raw_xy *= float(args.length / raw_arc[-1])
        raw_arc *= float(args.length / raw_arc[-1])
        target_arc = np.linspace(0.0, raw_arc[-1], steps)
        x = np.interp(target_arc, raw_arc, raw_xy[:, 0])
        y = np.interp(target_arc, raw_arc, raw_xy[:, 1])
    elif args.shape == "sharp_v_curves":
        # Alternating smooth cubic curves and deliberately sharp V vertices.
        # The corners are not rounded: this is a diagnostic for abrupt heading
        # changes, unlike the S-curve and circle tests.
        samples_per_segment = 70

        def line(start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
            return np.stack(
                (
                    np.linspace(start[0], end[0], samples_per_segment),
                    np.linspace(start[1], end[1], samples_per_segment),
                ),
                axis=1,
            )

        def bezier(
            p0: tuple[float, float],
            p1: tuple[float, float],
            p2: tuple[float, float],
            p3: tuple[float, float],
        ) -> np.ndarray:
            t = np.linspace(0.0, 1.0, samples_per_segment)[:, None]
            one_minus_t = 1.0 - t
            points = (
                one_minus_t**3 * np.asarray(p0)
                + 3.0 * one_minus_t**2 * t * np.asarray(p1)
                + 3.0 * one_minus_t * t**2 * np.asarray(p2)
                + t**3 * np.asarray(p3)
            )
            return points.astype(np.float32)

        # Geometry in the local robot frame.  p2->p3->p4 and
        # p5->p6->p7 are intentionally unrounded V-shaped turns.
        raw_xy = np.concatenate(
            (
                line((0.0, 0.0), (0.75, 0.0)),
                bezier((0.75, 0.0), (0.95, 0.0), (1.25, 0.45), (1.55, 0.45))[1:],
                line((1.55, 0.45), (2.15, 1.05))[1:],
                line((2.15, 1.05), (1.35, 1.65))[1:],
                bezier((1.35, 1.65), (1.0, 1.65), (0.55, 1.25), (0.35, 1.55))[1:],
                line((0.35, 1.55), (-0.35, 2.15))[1:],
                line((-0.35, 2.15), (-1.2, 1.7))[1:],
                bezier((-1.2, 1.7), (-1.3, 1.45), (-1.4, 0.8), (-1.45, 0.5))[1:],
            ),
            axis=0,
        )
        raw_arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(raw_xy, axis=0), axis=1)))
        )
        scale = float(args.length / raw_arc[-1])
        raw_xy *= scale
        raw_arc *= scale
        target_arc = np.linspace(0.0, raw_arc[-1], steps)
        x = np.interp(target_arc, raw_arc, raw_xy[:, 0])
        y = np.interp(target_arc, raw_arc, raw_xy[:, 1])
    else:
        x = distance
        y = np.zeros_like(distance)

    xy = np.stack((x, y), axis=1).astype(np.float32)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    if args.height_cycle is not None:
        cycle = np.asarray(args.height_cycle, dtype=np.float32)
        if cycle.size < 2 or np.any((cycle < 0.10) | (cycle > 0.40)):
            raise ValueError("--height_cycle needs at least two heights in [0.10, 0.40].")
        segment_index = np.floor(arc / args.height_segment_m).astype(np.int64)
        heights = cycle[segment_index % len(cycle)]
        schedule_description = f"cycle={cycle.tolist()}, segment={args.height_segment_m:.2f}m"
    else:
        # Piecewise height agenda; smoothing is deliberately left to the policy.
        heights = np.where(distance < args.transition_x, args.z_start, args.z_end)
        schedule_description = f"single transition at x={args.transition_x:.2f}m"

    path_points = np.column_stack((xy, heights)).astype(np.float32)

    path_array = np.array(path_points, dtype=np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, path_array)
    print(f"[INFO] Generated {args.shape} path of length {args.length}m with {schedule_description}.")
    print(f"[INFO] Saved to: {output}")


if __name__ == "__main__":
    main()
