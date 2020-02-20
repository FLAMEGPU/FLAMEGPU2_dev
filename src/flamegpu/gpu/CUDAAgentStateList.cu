 /**
 * @file CUDAAgentStateList.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAAgentStateList.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/pop/AgentStateMemory.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/gpu/CUDAScatter.h"

/**
* CUDAAgentStateList class
* @brief populates CUDA agent map
*/
CUDAAgentStateList::CUDAAgentStateList(CUDAAgent& cuda_agent)
    : condition_state(0)
    , current_list_size(0)
    , agent(cuda_agent) {
    // allocate state lists
    allocateDeviceAgentList(d_list);
    allocateDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().isOutputOnDevice())
        allocateDeviceAgentList(d_new_list);
    // Init condition state lists
    for (const auto &c : d_list)
        condition_d_list.emplace(c);
    for (const auto &c : d_swap_list)
        condition_d_swap_list.emplace(c);
}

/**
 * A destructor.
 * @brief Destroys the CUDAAgentStateList object
 */
CUDAAgentStateList::~CUDAAgentStateList() {
    cleanupAllocatedData();
}

void CUDAAgentStateList::cleanupAllocatedData() {
    // clean up
    releaseDeviceAgentList(d_list);
    releaseDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().isOutputOnDevice()) {
        releaseDeviceAgentList(d_new_list);
    }
    condition_d_list.clear();
    condition_d_swap_list.clear();
}

void CUDAAgentStateList::resize() {
    resizeDeviceAgentList(d_list, true);
    resizeDeviceAgentList(d_swap_list, false);
    resizeDeviceAgentList(d_new_list, false);
    setConditionState(condition_state);  // Update pointers in condition state list
}
void CUDAAgentStateList::resizeDeviceAgentList(CUDAMemoryMap &agent_list, bool copyData) {
    const auto &mem = agent.getAgentDescription().variables;

    // For each variable
    for (const auto &mm : mem) {
        const std::string var_name = mm.first;
        const size_t &type_size = agent.getAgentDescription().variables.at(mm.first).type_size;
        const size_t alloc_size = type_size * agent.getMaximumListSize();
        {
            // Allocate bigger new memory
            void * old_ptr = agent_list.at(var_name);  // Exception thrown if map doesn't contain variables that it should
            void * new_ptr = nullptr;
#ifdef UNIFIED_GPU_MEMORY
            // unified memory allocation
            gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&new_ptr), alloc_size))
#else
            // non unified memory allocation
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&new_ptr), alloc_size));
#endif
            if (copyData) {
                const size_t active_len = current_list_size * type_size;
                const size_t inactive_len = (agent.getMaximumListSize() - current_list_size) * type_size;
                // Copy across old data
                gpuErrchk(cudaMemcpy(new_ptr, old_ptr, active_len, cudaMemcpyDeviceToDevice));
                // Zero remaining new data
                gpuErrchk(cudaMemset(reinterpret_cast<char*>(new_ptr) + active_len, 0, inactive_len));
            } else {
                // Zero remaining new data
                gpuErrchk(cudaMemset(new_ptr, 0, alloc_size));
            }
            // Release old data
            gpuErrchk(cudaFree(old_ptr));
            // Replace old data in class member vars
            auto it = agent_list.find(var_name);  // No null check, call to .at() above will throw exception
            it->second = new_ptr;
        }
    }
}
/**
* @brief Allocates Device agent list
* @param variable of type CUDAMemoryMap type
* @return none
*/
void CUDAAgentStateList::allocateDeviceAgentList(CUDAMemoryMap &memory_map) {
    // we use the agents memory map to iterate the agent variables and do allocation within our GPU hash map
    const auto &mem = agent.getAgentDescription().variables;

    // for each variable allocate a device array and add to map
    for (const auto &mm : mem) {
        // get the variable name
        std::string var_name = mm.first;

        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(mm.first).type_size;

        // do the device allocation
        void * d_ptr;

#ifdef UNIFIED_GPU_MEMORY
        // unified memory allocation
        gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&d_ptr), var_size * agent.getMaximumListSize()))
#else
        // non unified memory allocation
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_ptr), var_size * agent.getMaximumListSize()));
#endif

        // store the pointer in the map
        memory_map.insert(CUDAMemoryMap::value_type(var_name, d_ptr));
    }
}

/**
* @brief Frees
* @param variable of type CUDAMemoryMap struct type
* @return none
*/
void CUDAAgentStateList::releaseDeviceAgentList(CUDAMemoryMap& memory_map) {
    // for each device pointer in the cuda memory map we need to free these
    for (const CUDAMemoryMapPair& mm : memory_map) {
        // free the memory on the device
        gpuErrchk(cudaFree(mm.second));
    }
}

/**
* @brief
* @param variable of type CUDAMemoryMap struct type
* @return none
*/
void CUDAAgentStateList::zeroDeviceAgentList(CUDAMemoryMap& memory_map) {
    // for each device pointer in the cuda memory map set the values to 0
    for (const CUDAMemoryMapPair& mm : memory_map) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(mm.first).type_size;

        // set the memory to zero
        gpuErrchk(cudaMemset(mm.second, 0, var_size*agent.getMaximumListSize()));
    }
}

/**
* @brief
* @param AgenstStateMemory object
* @return none
* @todo
*/
void CUDAAgentStateList::setAgentData(const AgentStateMemory &state_memory) {
    // check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription())) {
        THROW InvalidCudaAgentDesc("Agent State memory has different description to CUDA Agent ('%s'), "
            "in CUDAAgentStateList::setAgentData().",
            agent.getAgentDescription().name.c_str());
    }

    // set the current list size
    current_list_size = state_memory.getStateListSize();

    // copy raw agent data to device pointers
    for (CUDAMemoryMapPair m : d_list) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(m.first).type_size;

        // get the vector
        const GenericMemoryVector &m_vec = state_memory.getReadOnlyMemoryVector(m.first);

        // get pointer to vector data
        const void * v_data = m_vec.getReadOnlyDataPtr();

        // copy the host data to the GPU
        gpuErrchk(cudaMemcpy(m.second, v_data, var_size*current_list_size, cudaMemcpyHostToDevice));
    }

    // Update condition state lists
    setConditionState(0);
}

void CUDAAgentStateList::getAgentData(AgentStateMemory &state_memory) {
    // check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription())) {
        THROW InvalidCudaAgentDesc("Agent State memory has different description to CUDA Agent ('%s'), "
            "in CUDAAgentStateList::getAgentData().",
            agent.getAgentDescription().name.c_str());
    }

    // copy raw agent data to device pointers
    for (CUDAMemoryMapPair m : d_list) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(m.first).type_size;

        // get the vector
        GenericMemoryVector &m_vec = state_memory.getMemoryVector(m.first);

        // get pointer to vector data
        void * v_data = m_vec.getDataPtr();

        // check  the current list size
        if (current_list_size > state_memory.getPopulationCapacity()) {
            THROW InvalidMemoryCapacity("Current GPU state list size (%u) exceeds the state memory available (%u), "
                "in CUDAAgentStateList::getAgentData()",
                current_list_size, state_memory.getPopulationCapacity());
        }
        // copy the GPU data to host
        gpuErrchk(cudaMemcpy(v_data, m.second, var_size*current_list_size, cudaMemcpyDeviceToHost));

        // set the new state list size
        state_memory.overrideStateListSize(current_list_size);
    }
}

void* CUDAAgentStateList::getAgentListVariablePointer(std::string variable_name) const {
    CUDAMemoryMap::const_iterator mm = condition_d_list.find(variable_name);
    if (mm == condition_d_list.end()) {
        // TODO: Error variable not found in agent state list
        return nullptr;
    }

    return mm->second;
}

void CUDAAgentStateList::zeroAgentData() {
    zeroDeviceAgentList(d_list);
    zeroDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().isOutputOnDevice())
        zeroDeviceAgentList(d_new_list);
}

// the actual number of agents in this state
unsigned int CUDAAgentStateList::getCUDAStateListSize() const {
    return current_list_size - condition_state;
}
unsigned int CUDAAgentStateList::getCUDATrueStateListSize() const {
    return current_list_size;
}
void CUDAAgentStateList::setCUDAStateListSize(const unsigned int &newCount) {
    current_list_size = condition_state + newCount;
}

__global__ void scatter_living_agents(
    size_t typeLen,
    char * const __restrict__ in,
    char * out,
    const unsigned int streamId) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    // if optional message is to be written
    if (flamegpu_internal::CUDAScanCompaction::ds_agent_configs[streamId].scan_flag[index] == 1) {
        int output_index = flamegpu_internal::CUDAScanCompaction::ds_agent_configs[streamId].position[index];
        memcpy(out + (output_index * typeLen), in + (index * typeLen), typeLen);
    }
}
unsigned int CUDAAgentStateList::scatter(const unsigned int &streamId, const unsigned int out_offset, const ScatterMode &mode) {
    CUDAScatter &scatter = CUDAScatter::getInstance(streamId);
    const unsigned int living_agents = scatter.scatter(
        CUDAScatter::Type::Agent,
        agent.getAgentDescription().variables,
        condition_d_list, condition_d_swap_list,
        current_list_size, out_offset, mode == FunctionCondition2);
    // Swap
    assert(living_agents <= agent.getMaximumListSize());
    if (mode == Death) {
        std::swap(d_list, d_swap_list);
        std::swap(condition_d_list, condition_d_swap_list);
        current_list_size = living_agents;
    } else if (mode == FunctionCondition2) {
        std::swap(d_list, d_swap_list);
        std::swap(condition_d_list, condition_d_swap_list);
    }
    return living_agents;
}


void CUDAAgentStateList::setConditionState(const unsigned int &disabledAgentCt) {
    assert(disabledAgentCt <= current_list_size);
    condition_state = disabledAgentCt;
    // update condition_d_list and condition_d_swap_list
    const auto &mem = agent.getAgentDescription().variables;
    // for each variable allocate a device array and add to map
    for (const auto &mm : mem) {
        condition_d_list.at(mm.first) = reinterpret_cast<char*>(d_list.at(mm.first)) + (disabledAgentCt * mm.second.type_size);
        condition_d_swap_list.at(mm.first) = reinterpret_cast<char*>(d_swap_list.at(mm.first)) + (disabledAgentCt * mm.second.type_size);
    }
}
