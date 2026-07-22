"""Versioned tensor contracts shared by the B1 environment and tests."""

PROPRIO_DIM = 30
GOAL_DIM = 12
RESIDUAL_ACTION_DIM = 12

# proprio + diffusion goal + proposed base action + previous residual
# + [vx, vy, wz, signed cross-track, remaining fraction, height error,
#    tangent-speed error, sin(yaw error), cos(yaw error)]
ROUTE_STATE_DIM = 9
OBSERVATION_DIM = PROPRIO_DIM + GOAL_DIM + RESIDUAL_ACTION_DIM * 2 + ROUTE_STATE_DIM

GOAL_SCHEMA = "hindsight_geometric_average12_v1"
POLICY_KIND = "spatial_hindsight_geometry_ddpm"

