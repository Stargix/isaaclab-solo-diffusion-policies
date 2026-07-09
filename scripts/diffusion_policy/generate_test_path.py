import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Generate 3D paths for Solo12 Diffusion Policy evaluation.")
    parser.add_argument("--shape", type=str, default="straight", choices=["straight", "s_curve"], help="Shape of the path.")
    parser.add_argument("--length", type=float, default=8.0, help="Total length of the path in meters.")
    parser.add_argument("--step_size", type=float, default=0.05, help="Step size between path points.")
    parser.add_argument("--z_start", type=float, default=0.24, help="Initial height (default: 0.24m).")
    parser.add_argument("--z_end", type=float, default=0.16, help="Target ending height (default: 0.16m).")
    parser.add_argument("--transition_x", type=float, default=3.0, help="X position where height transitions from z_start to z_end.")
    parser.add_argument("--output", type=str, default="scripts/diffusion_policy/paths/test_path.npy", help="Output file path (.npy).")
    args = parser.parse_args()

    steps = int(args.length / args.step_size)
    path_points = []

    for i in range(steps):
        dx = i * args.step_size
        
        # Geometry
        x = dx
        if args.shape == "s_curve":
            y = 0.2 * np.sin(2.0 * np.pi * dx / args.length)
        else:
            y = 0.0
            
        # Z Height profile
        if dx < args.transition_x:
            z = args.z_start
        else:
            z = args.z_end
            
        path_points.append([x, y, z])

    path_array = np.array(path_points, dtype=np.float32)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, path_array)
    print(f"[INFO] Generated {args.shape} path of length {args.length}m, transitioning at {args.transition_x}m from Z={args.z_start}m to Z={args.z_end}m.")
    print(f"[INFO] Saved to: {args.output}")

if __name__ == "__main__":
    main()
