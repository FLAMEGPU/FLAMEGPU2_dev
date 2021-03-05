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
    for (AgentVector::Agent ai : av) {
        ai.setVariable<int>("int", ai.getVariable<int>("int") + 12);
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

FLAMEGPU_STEP_FUNCTION(Resize) {
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
TEST(DeviceAgentVectorTest, Resize) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector works as expected (when CUDAAgent resizes)
    ModelDescription model(MODEL_NAME);
    AgentDescription& agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(Resize);

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
FLAMEGPU_STEP_FUNCTION(Insert) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    AgentInstance ai(av[0]);
    av.insert(av.size() - AGENT_COUNT/2, 1024, ai);
    agent.setPopulationData(av);
}
FLAMEGPU_STEP_FUNCTION(Erase) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    av.erase(av.size() - AGENT_COUNT / 4, AGENT_COUNT / 2);
    av.push_back();
    av.back().setVariable<int>("int", static_cast<int>(av.size()));
    agent.setPopulationData(av);
}
FLAMEGPU_EXIT_CONDITION(AlwaysExit) {
    return EXIT;
}
TEST(DeviceAgentVectorTest, SubmodelResize) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector works as expected
    // Specifically, that when the agent population is resized, unbound variabled in the master-model are default init
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription& sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(Resize);
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
    for (unsigned int i = 0; i < AGENT_COUNT * 2; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 3);
    for (unsigned int i = 0; i < AGENT_COUNT * 3; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
    }
}
TEST(DeviceAgentVectorTest, SubmodelInsert) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector::insert works as expected
    // Specifically, that when the agent population is resized, unbound variabled in the master-model are default init
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription& sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(Insert);
    sub_model.addExitCondition(AlwaysExit);


    ModelDescription master_model(MODEL_NAME);
    AgentDescription& master_agent = master_model.newAgent(AGENT_NAME);
    master_agent.newVariable<int>("int", 0);
    master_agent.newVariable<float>("float", 12.0f);
    SubModelDescription& sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
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
    for (unsigned int i = 0; i < AGENT_COUNT * 2; ++i) {
        if (i < AGENT_COUNT / 2) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        } else if (i < (2 * AGENT_COUNT) + (AGENT_COUNT / 2)) {
            // Clones of index 0 were inserted
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(0));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        } else {
            // Original were moved
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i - AGENT_COUNT));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        }
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 3);
    for (unsigned int i = 0; i < AGENT_COUNT * 3; ++i) {
        if (i < AGENT_COUNT / 2) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        }
        else if (i < (2 * AGENT_COUNT) + (AGENT_COUNT / 2)) {
            // Clones of index 0 were inserted
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(0));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        } else {
            // Original were moved
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i - (2 * AGENT_COUNT)));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        }
    }
}
TEST(DeviceAgentVectorTest, SubmodelErase) {
    // The intention of this test is to check that agent death via DeviceAgentVector::erase works as expected
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription& sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(Erase);
    sub_model.addExitCondition(AlwaysExit);


    ModelDescription master_model(MODEL_NAME);
    AgentDescription& master_agent = master_model.newAgent(AGENT_NAME);
    master_agent.newVariable<int>("int", 0);
    master_agent.newVariable<float>("float", 12.0f);
    SubModelDescription& sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
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
    ASSERT_EQ(av.size(), (AGENT_COUNT * 0.75) + 1);
    for (unsigned int i = 0; i < (AGENT_COUNT * 0.75) + 1; ++i) {
        if (i < AGENT_COUNT / 4) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        }
        else if (i < AGENT_COUNT / 2) {
            // Clones 500 agents were removed
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i - AGENT_COUNT / 2));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        }
        else {
            // 1 agent was inserted
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(0));
            ASSERT_EQ(av[i].getVariable<float>("float"), 12.0f);
        }
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
