#ifndef INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_

#include <string>
#include <utility>
#include <memory>
#include <list>

#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/gpu/CUDAFatAgentStateList.h"

class CUDAScatter;
class CUDAAgent;

/**
 * This class provides an AgentVector interface to agent data currently stored on the device during execution of a CUDASimulation
 *
 * It attempts to prevent unnecessary memory transfers, as copying all agent variable buffers to only use 1
 * Would result in a large number of redundant but costly memcpys between host and device
 */
class DeviceAgentVector : protected AgentVector {
 public:
    /**
      * Construct a DeviceAgentVector interface to the on-device data of cuda_agent
      * @param cuda_agent CUDAAgent instance holding pointers to the desired agent data
      * @param cuda_agent_state Name of the state within cuda_agent to represent.
      */
    DeviceAgentVector(CUDAAgent &cuda_agent, const std::string& cuda_agent_state, CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream);
    /**
     * Copy operations are disabled
     */
    // @todo SOLVE THIS
    // DeviceAgentVector(const DeviceAgentVector& other) = delete;
    // DeviceAgentVector& operator=(const DeviceAgentVector& other) = delete;
    /**
     * Copies changed agent data back to device
     */
    void syncChanges();

    // Expose AgentVector methods that work properly
    using AgentVector::at;
    using AgentVector::operator[];
    using AgentVector::front;
    using AgentVector::back;
    // using AgentVector::data; // Would need to assume whole vector changed

    using AgentVector::begin;
    using AgentVector::end;
    using AgentVector::cbegin;
    using AgentVector::cend;
    using AgentVector::rbegin;
    using AgentVector::rend;
    using AgentVector::crbegin;
    using AgentVector::crend;

    using AgentVector::empty;
    using AgentVector::size;
    /**
     * Returns the max theoretical size of this host vector
     * This does not reflect the device allocated buffer
     */
    using AgentVector::max_size;
    /**
     * Pre-allocates buffer space for this host vector
     * This does not affect the device allocated buffer, that is updated if necessary when agents are returned to device.
     */
    using AgentVector::reserve;
    /**
     * Returns the current capacity of the host vector
     * This does not reflect the capacity of the device allocated buffer
     */
    using AgentVector::capacity;
    /**
     * Reduces the current capacity to fit the size of the host vector
     * This does not affect the capacity of the device allocated buffer, that is updated if necessary when agents are returned to device.
     */
    using AgentVector::shrink_to_fit;
    using AgentVector::clear;

    using AgentVector::insert;  // Moves agents, will cause master-agent variables which are not bound to go out of sync.
#ifdef SWIG
    // using AgentVector::py_insert;  // Moves agents, will cause master-agent variables which are not bound to go out of sync.
#endif
    using AgentVector::erase;  // Moves agents, will cause master-agent variables which are not bound to go out of sync.
    using AgentVector::push_back;
    using AgentVector::pop_back;
    using AgentVector::resize;  // This exposes the private method of the same name too!
    // using AgentVector::swap; // This would essentially require replacing the entire on-device agent vector

 protected:
    /**
     * Triggered when insert() has been called
     */
     void _insert(size_type pos, size_type count) override;
     /**
      * Triggered when erase() has been called
      */
     void _erase(size_type pos, size_type count) override;
    /**
     * Pair of a host-backed device buffer
     * This allows transactions which impact master-agent unbound variables to work correctly
     */
    struct VariableBufferPair {
        /**
         * Never allocate
         */
        explicit VariableBufferPair(const std::shared_ptr<VariableBuffer> &_device)
          : device(_device) { }
        VariableBufferPair(VariableBufferPair&& other) {
            *this = std::move(other);
        }
        VariableBufferPair& operator=(VariableBufferPair&& other) {
            std::swap(this->host, other.host);
            std::swap(this->device, other.device);
            return *this;
        }
        /**
          * Copy operations are disabled
          */
        VariableBufferPair(const VariableBufferPair& other) = delete;
        VariableBufferPair& operator=(const VariableBufferPair& other) = delete;

        /**
         * Free host if
         */
        ~VariableBufferPair() {
            if (host) free(host);
        }
        /**
         * nullptr until required to be allocated
         */
        char *host = nullptr;
        /**/
        std::shared_ptr<VariableBuffer> device;
    };
    /**
     * Any operations which move agents just be applied to this buffers too
     */
    std::list<VariableBufferPair> unbound_buffers;
    /**
     * The currently known size of the device buffer
     * This is used to track size before the unbound_buffers are init
     * Can't use _size in place of this, as calls to insert/erase/etc update that before we are notified
     */
    unsigned int known_device_buffer_size = 0;
    /**
     * Number of agents currently allocated inside the host_buffers
     * Useful if the device buffers are resized via the CUDAAgent
     */
    unsigned int unbound_host_buffer_size = 0;
    unsigned int unbound_host_buffer_capacity = 0;
    /**
     * Initialises the host copies of the unbound buffers
     * Allocates the host copy, and copies device data to them
     */
    void initUnboundBuffers();
    /**
     * Resizes the host copy of the unbound buffers, retaining data
     * @param new_capacity New buffer capacity
     * @param init If true, new memory is init
     */
    void resizeUnboundBuffers(const unsigned int & new_capacity, bool init);
    CUDAAgent &cuda_agent;
    std::string cuda_agent_state;

    CUDAScatter& scatter;
    const unsigned int& streamId;
    const cudaStream_t& stream;
};

#endif  // INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_
