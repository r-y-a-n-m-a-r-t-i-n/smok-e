from pydrake.multibody.plant import MultibodyPlant
import pydot
import numpy as np
from IPython.display import SVG, display

from pydrake.all import Simulator, DiagramBuilder, AddMultibodyPlantSceneGraph,\
                        Parser, RigidTransform, MeshcatVisualizer, MeshcatVisualizerParams, \
                        ConstantVectorSource, ConstantValueSource, PiecewisePolynomial,\
                        AbstractValue, HalfSpace, CoulombFriction


def simulate(sol):

    print("sim begun")

    plant = MultibodyPlant(0.0)
    parser = Parser(plant)
    parser.AddModelFromFile("arm.urdf")
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetBodyByName("base_z").body_frame(),
        RigidTransform.Identity()
    )
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()

    print("urdf imported + parsed")

    # Build the block diagram for the simulation
    builder = DiagramBuilder()

    # Add a planar walker to the simulation
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0005)
    X_WG = HalfSpace.MakePose(np.array([0, 0, 1]), np.zeros(3, ))
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_WG, HalfSpace(),
        "collision",
        CoulombFriction(1.0, 1.0)
    )

    print("added URDF to scene")

    # TODO: Adjust target moving speed here
    travel_speed = 0.5  # speed in m/s

    planner = builder.AddSystem()
    speed_src = builder.AddSystem(ConstantVectorSource(np.array([travel_speed])))
    base_traj_src = builder.AddSystem(
        ConstantValueSource(AbstractValue.Make(PiecewisePolynomial(np.zeros(1, ))))
    )

    # Wire planner inputs
    builder.Connect(plant.get_state_output_port(),
                    planner.get_state_input_port())
    builder.Connect(speed_src.get_output_port(),
                    planner.get_walking_speed_input_port())

    # Add the visualizer
    vis_params = MeshcatVisualizerParams(publish_period=0.01)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)

    # simulate
    diagram = builder.Build()
    display(SVG(pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2))[0].create_svg()))

    print("started simulating")

    sim_time = 10.0
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1)

    # Set the robot state
    plant_context = diagram.GetMutableSubsystemContext(
        plant, simulator.get_mutable_context())
    q = np.zeros((plant.num_positions(),))
    q[0] = 0.0
    q[1] = np.pi / 2.0
    q[2] = 0.0
    plant.SetPositions(plant_context, q)

    # Simulate the robot
    simulator.AdvanceTo(sim_time)