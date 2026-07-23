"""Versioned tensor contracts shared by the B1 environment and tests."""

PROPRIO_DIM = 30
GOAL_DIM = 12
RESIDUAL_ACTION_DIM = 12

# proprio + diffusion goal + proposed base action + previous residual
# + [vx, vy, wz, signed cross-track, remaining fraction, height error,
#    tangent-speed error, sin(yaw error), cos(yaw error),
#    schedule error, mean-progress-speed error]
#
# The final two features make the average-speed objective Markov for the
# feed-forward PPO actor.  Checkpoints trained with the former 75D contract are
# intentionally incompatible instead of being loaded with changed semantics.
ROUTE_STATE_DIM = 11
OBSERVATION_DIM = PROPRIO_DIM + GOAL_DIM + RESIDUAL_ACTION_DIM * 2 + ROUTE_STATE_DIM

GOAL_SCHEMA = "hindsight_geometric_average12_v1"
OBSERVATION_SCHEMA = "residual_route_state77_average_speed_v2"
POLICY_KIND = "spatial_hindsight_geometry_ddpm"
