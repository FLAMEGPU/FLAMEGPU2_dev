#ifndef INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_

#include <string>

#include "flamegpu/pop/AgentVector.h"

class CUDAScatter;
class CUDAAgent;

/**
 * This class provides an AgentVector interface to agent data currently stored on the device
 *
 * It attempts to prevent unnecessary memory transfers, as copying all agent variable buffers to only use 1
 * Would result in a large number of redundant but cost memcpys between host and device
 */
class DeviceAgentVector : protected AgentVector {
 public:
    /**
      * Construct a DeviceAgentVector interface to the on-device data of cuda_agent
      * @param cuda_agent CUDAAgent instnce holding pointers to the desired agent data
      * @param cuda_agent_state Name of the state within cuda_agent to represent.
      */
    DeviceAgentVector(CUDAAgent &cuda_agent, const std::string& cuda_agent_state);
    /**
     * Copies changed agent data back to device
     */
    void syncChanges(CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream);

    // Expose AgentVector methods that work properly
    using AgentVector::operator[];
    using AgentVector::at;
    using AgentVector::begin;
    using AgentVector::end;
    using AgentVector::cbegin;
    using AgentVector::cend;
    using AgentVector::rbegin;
    using AgentVector::rend;
    using AgentVector::crbegin;
    using AgentVector::crend;
    using AgentVector::resize;  // This exposes the private method of the same name too!
    using AgentVector::size;

 private:
  CUDAAgent &cuda_agent;
  std::string cuda_agent_state;
};

#endif  // INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_
