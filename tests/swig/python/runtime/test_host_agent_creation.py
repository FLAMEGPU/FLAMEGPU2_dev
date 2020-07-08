import pytest
from unittest import TestCase
from pyflamegpu import *

AGENT_COUNT = 1024
INIT_AGENT_COUNT = 512
NEW_AGENT_COUNT = 512

class BasicOutput(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT):
            FLAMEGPU.newAgent("agent").setVariableFloat("x", 1.0)
     
class BasicOutputCdn(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT):
            FLAMEGPU.newAgent("agent").setVariableFloat("x", 1.0)
        return pyflamegpu.CONTINUE  # New agents wont be created if EXIT is passed
    
class OutputState(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT):
            FLAMEGPU.newAgent("agent", "b").setVariableFloat("x", 1.0)

class OutputMultiAgent(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            FLAMEGPU.newAgent("agent", "b").setVariableFloat("x", 1.0)
            FLAMEGPU.newAgent("agent2").setVariableFloat("y", 2.0)
        
class BadVarName(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent").setVariableFloat("nope", 1.0)

class BadVarType(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent").setVariableInt64("x", 1.0)
        
class Getter(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            newAgt = FLAMEGPU.newAgent("agent")
            newAgt.setVariableFloat("x", newAgt.getVariableFloat("default"))
        
class GetBadVarName(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            newAgt = FLAMEGPU.newAgent("agent")
            FLAMEGPU.newAgent("agent").getVariableFloat("nope")
                
class GetBadVarType(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            newAgt = FLAMEGPU.newAgent("agent")
            FLAMEGPU.newAgent("agent").getVariableInt64("x")
        
class ArrayVarHostBirth(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(AGENT_COUNT): 
            a = FLAMEGPU.newAgent("agent_name")
            a.setVariableUInt("id", i)
            a.setVariableIntArray4("array_var",  2 + i, 4 + i, 8 + i, 16 + i )
            a.setVariableInt("array_var2", 0, 3 + i)
            a.setVariableInt("array_var2", 1, 5 + i)
            a.setVariableInt("array_var2", 2, 9 + i)
            a.setVariableInt("array_var2", 3, 17 + i)
            a.setVariableFloat("y", 14.0 + i)

class ArrayVarHostBirthSetGet(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(AGENT_COUNT): 
            a = FLAMEGPU.newAgent("agent_name")
            a.setVariableUInt("id", i)
            # Set
            a.setVariableIntArray4("array_var",  2 + i, 4 + i, 8 + i, 16 + i )
            a.setVariableInt("array_var2", 0, 3 + i)
            a.setVariableInt("array_var2", 1, 5 + i)
            a.setVariableInt("array_var2", 2, 9 + i)
            a.setVariableInt("array_var2", 3, 17 + i)
            a.setVariableFloat("y", 14.0 + i)
            # GetSet
            a.setVariableIntArray4("array_var", a.getVariableIntArray4("array_var"))
            a.setVariableInt("array_var2", 0, a.getVariableInt("array_var2", 0))
            a.setVariableInt("array_var2", 1, a.getVariableInt("array_var2", 1))
            a.setVariableInt("array_var2", 2, a.getVariableInt("array_var2", 2))
            a.setVariableInt("array_var2", 3, a.getVariableInt("array_var2", 3))
            a.setVariableFloat("y", a.getVariableFloat("y"))
        
class ArrayVarHostBirth_DefaultWorks(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(AGENT_COUNT): 
            FLAMEGPU.newAgent("agent_name")
               
class ArrayVarHostBirth_LenWrong(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableArray8("array_var", [0]*4)
        
class ArrayVarHostBirth_LenWrong2(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("array_var", 5, 0)
        
class ArrayVarHostBirth_TypeWrong(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableFloatArray4("array_var", [0]*4)
        
class ArrayVarHostBirth_TypeWrong2(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableFloat("array_var", 4, 0.0)
        
class ArrayVarHostBirth_NameWrong(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableIntArray4("array_varAAAAAA", [0]*4)
        
class ArrayVarHostBirth_NameWrong2(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("array_varAAAAAA", 4, 0)
        
class ArrayVarHostBirth_ArrayNotSuitableSet(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("array_var", 12)
        
class ArrayVarHostBirth_ArrayNotSuitableSet(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").getVariableInt("array_var")
        
        
class HostAgentCreationTest(TestCase):


    def test_from_init(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.addInitFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_step(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_host_layer(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.newLayer().addHostFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_exit_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutputCdn()
        model.addExitConditionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_step_empty_pop(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"))
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == NEW_AGENT_COUNT
        is_1 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
        
        assert is_1 == NEW_AGENT_COUNT

    def test_from_step_multi_state(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        func = OutputState()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance("a")
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize("a") == INIT_AGENT_COUNT
        assert population.getCurrentListSize("b") == NEW_AGENT_COUNT
        for i in range (population.getCurrentListSize("a")): 
            ai = population.getInstanceAt(i, "a")
            assert 12.0 == ai.getVariableFloat("x")
        
        for i in range (population.getCurrentListSize("b")): 
            ai = population.getInstanceAt(i, "b")
            assert 1.0 == ai.getVariableFloat("x")
        
"""
    TEST(HostAgentCreationTest, FromStepMultiAgent) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent2 = model.newAgent("agent2")
        agent2.newVariableFloat("y")
        model.addStepFunctionCallback(self.OutputMultiAgent)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance("a")
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        population = pyflamegpu.AgentPopulation2(agent2)
        cuda_model.getPopulationData(population2)
        # Validate each agent has same result
        assert population.getCurrentListSize("a") == INIT_AGENT_COUNT
        assert population.getCurrentListSize("b") == NEW_AGENT_COUNT
        assert population2.getCurrentListSize() == NEW_AGENT_COUNT
        for (i = 0 i < population.getCurrentListSize("a")  += 1i) 
            ai = population.getInstanceAt(i, "a")
            assert 12.0 == ai.getVariableFloat("x")
        
        for (i = 0 i < population.getCurrentListSize("b")  += 1i) 
            ai = population.getInstanceAt(i, "b")
            assert 1.0 == ai.getVariableFloat("x")
        
        for (i = 0 i < population2.getCurrentListSize()  += 1i) 
            ai = population2.getInstanceAt(i)
            assert 2.0 == ai.getVariableFloat("y")
        

    TEST(HostAgentCreationTest, DefaultVariableValue) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        model.addStepFunctionCallback(self.BasicOutput)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), NEW_AGENT_COUNT)
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() ==  NEW_AGENT_COUNT
        is_15 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("default")
            if val == 15.0:
                is_15 += 1
        
        assert is_15 == NEW_AGENT_COUNT

    TEST(HostAgentCreationTest, BadVarName) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        model.addStepFunctionCallback(self.BadVarName)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        # Execute model
        EXPECT_THROW(cuda_model.step(), InvalidAgentVar)

    TEST(HostAgentCreationTest, BadVarType) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        model.addStepFunctionCallback(self.BadVarType)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        # Execute model
        EXPECT_THROW(cuda_model.step(), InvalidVarType)

    TEST(HostAgentCreationTest, GetterWorks) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        model.addStepFunctionCallback(self.Getter)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), NEW_AGENT_COUNT)
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == NEW_AGENT_COUNT
        is_15 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 15.0:
                is_15 += 1
        
        # Every host created agent has had their default loaded from "default" and stored in "x"
        assert is_15 == NEW_AGENT_COUNT

    TEST(HostAgentCreationTest, GetterBadVarName) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        model.addStepFunctionCallback(self.GetBadVarName)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        # Execute model
        EXPECT_THROW(cuda_model.step(), InvalidAgentVar)

    TEST(HostAgentCreationTest, GetterBadVarType) 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        model.addStepFunctionCallback(self.GetBadVarType)
        # Init agent pop
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        # Execute model
        EXPECT_THROW(cuda_model.step(), InvalidVarType)


    TEST(HostAgentCreationTest, HostAgentBirth_ArraySet) 
        const std::arrayIntArray4 TEST_REFERENCE =  2, 4, 8, 16 
        const std::arrayIntArray4 TEST_REFERENCE2 =  3, 5, 9, 17 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id", UINT_MAX)
        agent.newVariableIntArray4("array_var")
        agent.newVariableIntArray4("array_var2")
        agent.newVariableFloat("y", 13.0)
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth)
        CUDAAgentModel sim(model)
        sim.step()
        population = pyflamegpu.AgentPopulation(agent)
        sim.getPopulationData(population)
        # Check data is correct
        assert population.getCurrentListSize() == AGENT_COUNT
        for (i = 0 i < population.getCurrentListSize() i += 1) 
            instance = population.getInstanceAt(i)
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = instance.getVariableIntArray4("array_var")
            array2 = instance.getVariableIntArray4("array_var2")
            for (k = 0 k < 4  += 1k) 
                array1[k] -= j
                array2[k] -= j
            
            assert array1 == TEST_REFERENCE
            assert array2 == TEST_REFERENCE2
            assert instance.getVariableFloat("y") == 14 + j
        

    TEST(HostAgentCreationTest, HostAgentBirth_ArraySetGet) 
        const std::arrayIntArray4 TEST_REFERENCE =  2, 4, 8, 16 
        const std::arrayIntArray4 TEST_REFERENCE2 =  3, 5, 9, 17 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id", UINT_MAX)
        agent.newVariableIntArray4("array_var")
        agent.newVariableIntArray4("array_var2")
        agent.newVariableFloat("y", 13.0)
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirthSetGet)
        CUDAAgentModel sim(model)
        sim.step()
        population = pyflamegpu.AgentPopulation(agent)
        sim.getPopulationData(population)
        # Check data is correct
        assert population.getCurrentListSize() == AGENT_COUNT
        for (i = 0 i < population.getCurrentListSize() i += 1) 
            instance = population.getInstanceAt(i)
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = instance.getVariableIntArray4("array_var")
            array2 = instance.getVariableIntArray4("array_var2")
            for (k = 0 k < 4  += 1k) 
                array1[k] -= j
                array2[k] -= j
            
            assert array1 == TEST_REFERENCE
            assert array2 == TEST_REFERENCE2
            assert instance.getVariableFloat("y") == 14 + j
        

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayDefaultWorks) 
        const std::arrayIntArray4 TEST_REFERENCE =  2, 4, 8, 16 
        const std::arrayIntArray4 TEST_REFERENCE2 =  3, 5, 9, 17 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id", UINT_MAX)
        agent.newVariableIntArray4("array_var", TEST_REFERENCE)
        agent.newVariableIntArray4("array_var2", TEST_REFERENCE2)
        agent.newVariableFloat("y", 13.0)
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_DefaultWorks)
        CUDAAgentModel sim(model)
        sim.step()
        population = pyflamegpu.AgentPopulation(agent)
        sim.getPopulationData(population)
        # Check data is correct
        assert population.getCurrentListSize() == AGENT_COUNT
        for (i = 0 i < population.getCurrentListSize() i += 1) 
            instance = population.getInstanceAt(i)
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = instance.getVariableIntArray4("array_var")
            array2 = instance.getVariableIntArray4("array_var2")
            assert instance.getVariableUInt("id") == UINT_MAX
            assert array1 == TEST_REFERENCE
            assert array2 == TEST_REFERENCE2
            assert instance.getVariableFloat("y") == 13.0
        

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_LenWrong)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidVarArrayLen)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong2) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_LenWrong2)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), OutOfRangeVarArray)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong3) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableInt("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_LenWrong)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidVarArrayLen)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong4) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableInt("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_LenWrong2)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), OutOfRangeVarArray)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayTypeWrong) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_TypeWrong)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidVarType)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayTypeWrong2) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_TypeWrong2)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidVarType)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayNameWrong) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_NameWrong)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidAgentVar)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayNameWrong2) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_NameWrong)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidAgentVar)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayNotSuitableSet) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_ArrayNotSuitableSet)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidAgentVar)

    TEST(HostAgentCreationTest, HostAgentBirth_ArrayNotSuitableGet) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        model.addStepFunctionCallback(self.ArrayVarHostBirth_ArrayNotSuitableGet)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), InvalidAgentVar)

    FLAMEGPU_STEP_FUNCTION(reserved_name_step) 
        FLAMEGPU.newAgent("agent_name").setVariableInt("_", 0)

    FLAMEGPU_STEP_FUNCTION(reserved_name_step_array) 
        FLAMEGPU.newAgent("agent_name").setVariable<int, 3>("_", )

    TEST(HostAgentCreationTest, reserved_name) 
        model = pyflamegpu.ModelDescription("model")
        model.newAgent("agent_name")
        # Run the init function
        model.addStepFunctionCallback(self.reserved_name_step)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), ReservedName)

    TEST(HostAgentCreationTest, reserved_name_array) 
        model = pyflamegpu.ModelDescription("model")
        model.newAgent("agent_name")
        model.addStepFunctionCallback(self.reserved_name_step_array)
        CUDAAgentModel sim(model)
        EXPECT_THROW(sim.step(), ReservedName)

"""
