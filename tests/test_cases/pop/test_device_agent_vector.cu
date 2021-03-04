#include <string>

#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"

namespace DeviceAgentVectorTest {
    const unsigned int AGENT_COUNT = 1024;
    const std::string MODEL_NAME = "model";
    const std::string SUBMODEL_NAME = "submodel";
    const std::string AGENT_NAME = "agent";

FLAMEGPU_STEP_FUNCTION(SetGet) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    for (AgentVector::Agent agent : av) {
        agent.setVariable<int>("int", agent.getVariable<int>("int") + 12);
    }
    agent.setPopulationData(av);
}

TEST(DeviceAgentVectorTest, SetGet) {
    // Initialise an agent population with values in a variable [0,1,2..N]
    // Inside a step function, retrieve the agent population as a DeviceAgentVector
    // Update all agents by adding 12 to their value
    // After model completion, retrieve the agent population and check their values are [12,13,14..N+12]
    ModelDescription model(MODEL_NAME);
    AgentDescription& agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(SetGet);

    // Init agent pop
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
      av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 12);
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 24);
    }
}

FLAMEGPU_STEP_FUNCTION(New) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    unsigned int av_size = av.size();
    av.resize(av_size + AGENT_COUNT);
    // Continue the existing variable pattern
    for (unsigned int i = av_size; i < av_size + AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", i);
    }
    agent.setPopulationData(av);
}
TEST(DeviceAgentVectorTest, New) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector works as expected (when CUDAAgent resizes)
    ModelDescription model(MODEL_NAME);
    AgentDescription& agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(New);

    // Init agent pop
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
        av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 2);
    for (unsigned int i = 0; i < AGENT_COUNT * 2; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), i);
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 3);
    for (unsigned int i = 0; i < AGENT_COUNT * 3; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), i);
    }
}
FLAMEGPU_EXIT_CONDITION(AlwaysExit) {
    return EXIT;
}
TEST(DeviceAgentVectorTest, SubmodelNew) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector works as expected
    // Specifically, that when the agent population is resized, unbound variabled in the master-model are default init
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription& sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(New);
    sub_model.addExitCondition(AlwaysExit);


    ModelDescription master_model(MODEL_NAME);
    AgentDescription& master_agent = master_model.newAgent(AGENT_NAME);
    master_agent.newVariable<int>("int", 0);
    master_agent.newVariable<float>("float", 12.0f);
    SubModelDescription &sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop
    AgentVector av(master_agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
        av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 2);
    // Separate the loops for agents created at each stage to easier identify where the error is coming from
    for (unsigned int i = 0; i < AGENT_COUNT * 2; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 3);
    // Separate the loops for agents created at each stage to easier identify where the error is coming from
    for (unsigned int i = 0; i < AGENT_COUNT * 3; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
    }
}

}  // namespace DeviceAgentVectorTest
