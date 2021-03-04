#include "flamegpu/pop/DeviceAgentVector.h"
#include "flamegpu/gpu/CUDAAgent.h"

DeviceAgentVector::DeviceAgentVector(CUDAAgent& _cuda_agent, const std::string &_cuda_agent_state)
    : AgentVector(_cuda_agent.getAgentDescription(), 0)
    , cuda_agent(_cuda_agent)
    , cuda_agent_state(_cuda_agent_state) {
    // Create an empty AgentVector and initialise it manually
    // For each variable create an uninitialised array of variable data
    _size = cuda_agent.getStateSize(cuda_agent_state);
    resize(_size, false);
    // @todo Delay this process until individual variable data is requested?
    for (const auto& v : agent->variables) {
        // Copy back variable data into each array
        void *host_dest = _data->at(v.first)->getDataPtr();
        const void *device_src = cuda_agent.getStateVariablePtr(cuda_agent_state, v.first);
        gpuErrchk(cudaMemcpy(host_dest, device_src, _size * v.second.type_size * v.second.elements, cudaMemcpyDeviceToHost));
        // Default-init remaining buffer space (Not currently used as earlier resize call is exact)
        if (_capacity > _size) {
            init(_size, _capacity);
        }
    }
}

void DeviceAgentVector::syncChanges(CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream) {
    // Resize device buffers if necessary
    const unsigned int old_allocated_size = cuda_agent.getStateAllocatedSize(cuda_agent_state);
    if (_size > old_allocated_size) {
        const unsigned int old_size = cuda_agent.getStateSize(cuda_agent_state);
        // Resize the underlying variable buffers for this agent state and retain variable data
        cuda_agent.resizeState(cuda_agent_state, _size, true);  // @todo Don't retain data for mapped buffers?
        // Init agent data for any variables of newly created agents which are only present in a parent model
        const unsigned int new_allocated_size = cuda_agent.getStateAllocatedSize(cuda_agent_state);
        // This call does not use streams properly internally
        cuda_agent.initExcludedVars(cuda_agent_state, new_allocated_size - old_size, old_size, scatter, streamId, stream);
    }
    // Copy all changes back to device
    for (const auto& v : agent->variables) {
        // @todo Only copy changed data
        // Copy back variable data into each array
        const void* host_src = _data->at(v.first)->getDataPtr();
        void* device_dest = cuda_agent.getStateVariablePtr(cuda_agent_state, v.first);
        gpuErrchk(cudaMemcpyAsync(device_dest, host_src, _size * v.second.type_size * v.second.elements, cudaMemcpyHostToDevice, stream));
    }
    cudaStreamSynchronize(stream);
    // Update CUDAAgent statelist size
    cuda_agent.setStateAgentCount(cuda_agent_state, _size);
}
