#include <string>

#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"

namespace DeviceAgentVectorTest {
    const unsigned int AGENT_COUNT = 10;
    const std::string MODEL_NAME = "model";
    const std::string SUBMODEL_NAME = "submodel";
    const std::string AGENT_NAME = "agent";

FLAMEGPU_STEP_FUNCTION(SetGet) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    for (AgentVector::Agent ai : av) {
        ai.setVariable<int>("int", ai.getVariable<int>("int") + 12);
    }
}
FLAMEGPU_STEP_FUNCTION(SetGetHalf) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    for (unsigned int i = av.size()/4; i < av.size() - av.size()/4; ++i) {
        av[i].setVariable<int>("int", av[i].getVariable<int>("int") + 12);
    }
    // agent.setPopulationData(av);
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
TEST(DeviceAgentVectorTest, SetGetHalf) {
    // Initialise an agent population with values in a variable [0,1,2..N]
    // Inside a step function, retrieve the agent population as a DeviceAgentVector
    // Update half agents (contiguous block) by adding 12 to their value
    // After model completion, retrieve the agent population and check their values are [12,13,14..N+12]
    ModelDescription model(MODEL_NAME);
    AgentDescription& agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(SetGetHalf);

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
    ASSERT_EQ(av.size(), AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        if (i < AGENT_COUNT/4 || i >= AGENT_COUNT - AGENT_COUNT/4) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        } else  {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 12);
        }
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        if (i < AGENT_COUNT / 4 || i >= AGENT_COUNT - AGENT_COUNT / 4) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        } else {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 24);
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(MasterIncrement, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<unsigned int>("uint", FLAMEGPU->getVariable<unsigned int>("uint") + 1);
    return ALIVE;
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
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 3);
    for (unsigned int i = 0; i < AGENT_COUNT * 3; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
    }
}
FLAMEGPU_STEP_FUNCTION(Insert) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    AgentInstance ai(av[0]);
    av.insert(av.size() - AGENT_COUNT/2, AGENT_COUNT, ai);
}
FLAMEGPU_STEP_FUNCTION(Erase) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    av.erase(AGENT_COUNT / 4, AGENT_COUNT / 2);
    av.push_back();
    av.back().setVariable<int>("int", -2);
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
    master_agent.newVariable<unsigned int>("uint", 12u);
    master_agent.newFunction("MasterIncrement", MasterIncrement);
    SubModelDescription &sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addAgentFunction(MasterIncrement);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop
    AgentVector av(master_agent, AGENT_COUNT);
    std::vector<int> vec_int;
    std::vector<unsigned int> vec_uint;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
        av[i].setVariable<unsigned int>("uint", static_cast<int>(i));
        vec_int.push_back(i);
        vec_uint.push_back(i);
    }

    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.resize(vec_int.size() + AGENT_COUNT, 0);
    vec_uint.resize(vec_uint.size() + AGENT_COUNT, 12u);
    for (unsigned int i = AGENT_COUNT; i < 2 * AGENT_COUNT; ++i)
      vec_int[i] = static_cast<int>(i);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
    }

    // Step again
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.resize(vec_int.size() + AGENT_COUNT, 0);
    vec_uint.resize(vec_uint.size() + AGENT_COUNT, 12u);
    for (unsigned int i = 2 * AGENT_COUNT; i < 3 *AGENT_COUNT; ++i)
        vec_int[i] = static_cast<int>(i);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
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
    master_agent.newVariable<unsigned int>("uint", 12u);
    master_agent.newFunction("MasterIncrement", MasterIncrement);
    SubModelDescription& sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addAgentFunction(MasterIncrement);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop
    AgentVector av(master_agent, AGENT_COUNT);
    std::vector<int> vec_int;
    std::vector<unsigned int> vec_uint;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
        av[i].setVariable<unsigned int>("uint", static_cast<int>(i));
        vec_int.push_back(i);
        vec_uint.push_back(i);
    }

    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.insert(vec_int.begin() + (vec_int.size() - AGENT_COUNT / 2), AGENT_COUNT, vec_int[0]);
    vec_uint.insert(vec_uint.begin() + (vec_uint.size() - AGENT_COUNT / 2), AGENT_COUNT, 12u);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
    }

    // Step again
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.insert(vec_int.begin() + (vec_int.size() - AGENT_COUNT / 2), AGENT_COUNT, vec_int[0]);
    vec_uint.insert(vec_uint.begin() + (vec_uint.size() - AGENT_COUNT / 2), AGENT_COUNT, 12u);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
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
    master_agent.newVariable<int>("int", -1);
    master_agent.newVariable<float>("float", 12.0f);
    SubModelDescription& sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop, and test vectors
    AgentVector av(master_agent, AGENT_COUNT);
    std::vector<int> vec_int;
    std::vector<float> vec_flt;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
        vec_int.push_back(static_cast<int>(i));
        vec_flt.push_back(12.0f);
    }
    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();
    // Update vectors to match
    vec_int.erase(vec_int.begin() + (AGENT_COUNT / 4), vec_int.begin() + (AGENT_COUNT / 2));
    vec_flt.erase(vec_flt.begin() + (AGENT_COUNT /4), vec_flt.begin() + (AGENT_COUNT / 2));
    vec_int.push_back(-2);
    vec_flt.push_back(12.0f);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < vec_int.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<float>("float"), vec_flt[i]);
    }

    // Step again
    sim.step();
    // Update vectors to match
    vec_int.erase(vec_int.begin() + (AGENT_COUNT / 4), vec_int.begin() + (AGENT_COUNT / 2));
    vec_flt.erase(vec_flt.begin() + (AGENT_COUNT / 4), vec_flt.begin() + (AGENT_COUNT / 2));
    vec_int.push_back(-2);
    vec_flt.push_back(12.0f);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < vec_int.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<float>("float"), vec_flt[i]);
    }
}

}  // namespace DeviceAgentVectorTest
