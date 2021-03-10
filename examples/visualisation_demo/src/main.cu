#include "flamegpu/flame_api.h"
// Disable a warning caused by GLM using `__host__ __device__` and `= default` on functions
// Can't just wrap the include, as everywhere the affected functions are used triggers the warning.
#pragma diag_suppress = esa_on_defaulted_function_ignored
#include "../flamegpu2_visualiser-build/glm-src/glm/glm.hpp"  // Cheaty access to GLM

FLAMEGPU_AGENT_FUNCTION(direction2d_update, MsgNone, MsgNone) {
    glm::vec2 loc = glm::vec2(FLAMEGPU->getVariable<float>("location_x"), FLAMEGPU->getVariable<float>("location_y"));
    glm::vec2 vel = glm::vec2(FLAMEGPU->getVariable<float>("velocity_x"), FLAMEGPU->getVariable<float>("velocity_y"));
    // Add velocity to Location
    loc += vel;
    glm::vec2 env_min = glm::vec2(FLAMEGPU->environment.getProperty<float>("direction2d_envmin", 0), FLAMEGPU->environment.getProperty<float>("direction2d_envmin", 1));
    glm::vec2 env_max = glm::vec2(FLAMEGPU->environment.getProperty<float>("direction2d_envmax", 0), FLAMEGPU->environment.getProperty<float>("direction2d_envmax", 1));
    // Clamp agent to bounds (@todo Give agents a radius, rather than them being points)
    if (loc.x < env_min.x) {
        vel.x = -vel.x;
        loc.x = env_min.x + (env_min.x- loc.x);
    } else if (loc.x > env_max.x) {
        vel.x = -vel.x;
        loc.x = env_max.x - (loc.x - env_max.x);
    }
    if (loc.y < env_min.y) {
        vel.y = -vel.y;
        loc.y = env_min.y + (env_min.y - loc.y);
    } else if (loc.y > env_max.y) {
        vel.y = -vel.y;
        loc.y = env_max.y - (loc.y - env_max.y);
    }
    // Update values
    FLAMEGPU->setVariable<float>("location_x", loc.x);
    FLAMEGPU->setVariable<float>("location_y", loc.y);
    FLAMEGPU->setVariable<float>("velocity_x", vel.x);
    FLAMEGPU->setVariable<float>("velocity_y", vel.y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(direction3d_update, MsgNone, MsgNone) {
    glm::vec3 loc = glm::vec3(FLAMEGPU->getVariable<float>("location_x"), FLAMEGPU->getVariable<float>("location_y"), FLAMEGPU->getVariable<float>("location_z"));
    glm::vec3 vel = glm::vec3(FLAMEGPU->getVariable<float>("velocity_x"), FLAMEGPU->getVariable<float>("velocity_y"), FLAMEGPU->getVariable<float>("velocity_z"));
    // Add velocity to Location
    loc += vel;
    glm::vec3 env_min = glm::vec3(
        FLAMEGPU->environment.getProperty<float>("direction3d_envmin", 0),
        FLAMEGPU->environment.getProperty<float>("direction3d_envmin", 1),
        FLAMEGPU->environment.getProperty<float>("direction3d_envmin", 2));
    glm::vec3 env_max = glm::vec3(
        FLAMEGPU->environment.getProperty<float>("direction3d_envmax", 0),
        FLAMEGPU->environment.getProperty<float>("direction3d_envmax", 1),
        FLAMEGPU->environment.getProperty<float>("direction3d_envmax", 2));
    // Clamp agent to bounds (@todo Give agents a radius, rather than them being points)
    if (loc.x < env_min.x) {
        vel.x = -vel.x;
        loc.x = env_min.x + (env_min.x - loc.x);
    } else if (loc.x > env_max.x) {
        vel.x = -vel.x;
        loc.x = env_max.x - (loc.x - env_max.x);
    }
    if (loc.y < env_min.y) {
        vel.y = -vel.y;
        loc.y = env_min.y + (env_min.y - loc.y);
    } else if (loc.y > env_max.y) {
        vel.y = -vel.y;
        loc.y = env_max.y - (loc.y - env_max.y);
    }
    if (loc.z < env_min.z) {
        vel.z = -vel.z;
        loc.z = env_min.z + (env_min.z - loc.z);
    } else if (loc.z > env_max.z) {
        vel.z = -vel.z;
        loc.z = env_max.z - (loc.z - env_max.z);
    }
    // Update values
    FLAMEGPU->setVariable<float>("location_x", loc.x);
    FLAMEGPU->setVariable<float>("location_y", loc.y);
    FLAMEGPU->setVariable<float>("location_z", loc.z);
    FLAMEGPU->setVariable<float>("velocity_x", vel.x);
    FLAMEGPU->setVariable<float>("velocity_y", vel.y);
    FLAMEGPU->setVariable<float>("velocity_z", vel.z);
    return ALIVE;
}

int main(int argc, const char ** argv) {
    std::default_random_engine rng;

    ModelDescription model("Visualisation Demo");
    EnvironmentDescription &env = model.Environment();
    LayerDescription &layer1 = model.newLayer();

    // This model exists to demonstrate visualisation features
    // Each agent population demonstrates a different feature

    const int ENV_DIM = 50;
    const int ENV_GAP = 20;
    /**
     * Visualise an agent according to it's direction 2d
     */
    {
        // Direction agent 2d
        AgentDescription &agent = model.newAgent("direction2d");
        agent.newVariable<float>("location_x");
        agent.newVariable<float>("location_y");
        agent.newVariable<float>("velocity_x");
        agent.newVariable<float>("velocity_y");
        agent.newFunction("direction2d_update", direction2d_update);
        layer1.addAgentFunction(direction2d_update);
        // 50x50 plane
        env.newProperty<float, 2>("direction2d_envmin", std::array<float, 2>{0, 0});
        env.newProperty<float, 2>("direction2d_envmax", std::array<float, 2>{ENV_DIM, ENV_DIM});
    }
    AgentVector pop_direction2d(model.Agent("direction2d"), 5);
    {
        std::uniform_real_distribution<float> location_dist(0.0f, ENV_DIM);
        std::uniform_real_distribution<float> speed_dist(ENV_DIM / 600.0f, ENV_DIM / 300.0f);  // Cross the environment in 5-10 seconds
        std::uniform_real_distribution<float> direction_dist(-1.0f, 1.0f);
        // Scatter all agents in the square
        for (auto agent : pop_direction2d) {
            agent.setVariable<float>("location_x", location_dist(rng));
            agent.setVariable<float>("location_y", location_dist(rng));
            glm::vec2 velocity = normalize(glm::vec2(direction_dist(rng), direction_dist(rng))) * speed_dist(rng);
            agent.setVariable<float>("velocity_x", velocity.x);
            agent.setVariable<float>("velocity_y", velocity.y);
        }
    }

    /**
     * Visualise an agent according to it's direction 3d
     */
    {   // Direction agent 3d (@todo Convert agent vars to <float, 3>)
        AgentDescription &agent = model.newAgent("direction3d");
        agent.newVariable<float>("location_x");
        agent.newVariable<float>("location_y");
        agent.newVariable<float>("location_z");
        agent.newVariable<float>("velocity_x");
        agent.newVariable<float>("velocity_y");
        agent.newVariable<float>("velocity_z");
        agent.newFunction("direction3d_update", direction3d_update);
        layer1.addAgentFunction(direction3d_update);
        // 50x50x50 cube
        env.newProperty<float, 3>("direction3d_envmin", std::array<float, 3>{ENV_DIM + ENV_GAP, 0, 0});
        env.newProperty<float, 3>("direction3d_envmax", std::array<float, 3>{2 * ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM});
    }
    AgentVector pop_direction3d(model.Agent("direction3d"), 25);
    {
        std::uniform_real_distribution<float> location_dist(0.0f, ENV_DIM);
        std::uniform_real_distribution<float> speed_dist(ENV_DIM / 600.0f, ENV_DIM / 300.0f);  // Cross the environment in 5-10 seconds
        std::uniform_real_distribution<float> direction_dist(-1.0f, 1.0f);
        // Scatter all agents in the square
        for (auto agent : pop_direction3d) {
            agent.setVariable<float>("location_x", location_dist(rng));
            agent.setVariable<float>("location_y", location_dist(rng));
            agent.setVariable<float>("location_z", location_dist(rng));
            glm::vec3 velocity = normalize(glm::vec3(direction_dist(rng), direction_dist(rng), direction_dist(rng))) * speed_dist(rng);
            agent.setVariable<float>("velocity_x", velocity.x);
            agent.setVariable<float>("velocity_y", velocity.y);
            agent.setVariable<float>("velocity_z", velocity.z);
        }
    }


    /**
     * Create Model Runner
     */
    CUDASimulation cuda_model(model, argc, argv);
    cuda_model.setPopulationData(pop_direction2d);
    cuda_model.setPopulationData(pop_direction3d);

    /**
     * Create visualisation
     */
#ifdef VISUALISATION
    ModelVis &m_vis = cuda_model.getVisualisation();
    {
        m_vis.setInitialCameraLocation(6.28f, 39.9f, 94.8f);
        m_vis.setInitialCameraTarget(6.28f + 0.536f, 39.9f - 0.233f, 94.8f - 0.812f);
        m_vis.setCameraSpeed(0.02f);
        m_vis.setSimulationSpeed(50);
    }
    {
        auto & vis_agent = m_vis.addAgent("direction2d");
        vis_agent.setXVariable("location_x");
        vis_agent.setYVariable("location_y");
        // Position vars are named x, y so they are used by default
        vis_agent.setModel(Stock::Models::TEAPOT);
        vis_agent.setModelScale(ENV_DIM/10.0f);
        // Draw outline of environment boundary
        auto v_boundary = m_vis.newLineSketch(1.0f, 1.0f, 1.0f);
        v_boundary.addVertex(0, 0, 0); v_boundary.addVertex(0, ENV_DIM, 0);
        v_boundary.addVertex(0, 0, 0); v_boundary.addVertex(ENV_DIM, 0, 0);
        v_boundary.addVertex(ENV_DIM, 0, 0); v_boundary.addVertex(ENV_DIM, ENV_DIM, 0);
        v_boundary.addVertex(0, ENV_DIM, 0); v_boundary.addVertex(ENV_DIM, ENV_DIM, 0);
    }
    {
        auto& vis_agent = m_vis.addAgent("direction3d");
        vis_agent.setXVariable("location_x");
        vis_agent.setYVariable("location_y");
        vis_agent.setZVariable("location_z");
        // Position vars are named x, y, z so they are used by default
        vis_agent.setModel(Stock::Models::TEAPOT);
        vis_agent.setModelScale(ENV_DIM / 10.0f);
        // Draw outline of environment boundary
        auto v_boundary = m_vis.newLineSketch(1.0f, 1.0f, 1.0f);
        // Bottom square
        v_boundary.addVertex(ENV_DIM + ENV_GAP, 0, 0); v_boundary.addVertex(ENV_DIM + ENV_GAP, 0, ENV_DIM);
        v_boundary.addVertex(ENV_DIM + ENV_GAP, 0, 0); v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, 0, 0);
        v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, 0, ENV_DIM); v_boundary.addVertex(ENV_DIM + ENV_GAP, 0, ENV_DIM);
        v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, 0, ENV_DIM); v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, 0, 0);
        // Top square
        v_boundary.addVertex(ENV_DIM + ENV_GAP, ENV_DIM, 0); v_boundary.addVertex(ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM);
        v_boundary.addVertex(ENV_DIM + ENV_GAP, ENV_DIM, 0); v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, ENV_DIM, 0);
        v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM); v_boundary.addVertex(ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM);
        v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM); v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, ENV_DIM, 0);
        // Columns
        v_boundary.addVertex(ENV_DIM + ENV_GAP, 0, 0); v_boundary.addVertex(ENV_DIM + ENV_GAP, ENV_DIM, 0);
        v_boundary.addVertex(ENV_DIM + ENV_GAP, 0, ENV_DIM); v_boundary.addVertex(ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM);
        v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, 0, 0); v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, ENV_DIM, 0);
        v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, 0, ENV_DIM); v_boundary.addVertex(2 * ENV_DIM + ENV_GAP, ENV_DIM, ENV_DIM);
    }
    m_vis.activate();
#else
#error This example is a demonstration of visualisation capabilities, enable VISUALISATION in CMake to build it.
#endif

    /**
     * Execution
     */
    cuda_model.simulate();

#ifdef VISUALISATION
    m_vis.join();
#endif
    return 0;
}
