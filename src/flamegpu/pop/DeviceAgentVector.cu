#include "flamegpu/pop/DeviceAgentVector.h"
#include "flamegpu/gpu/CUDAAgent.h"

DeviceAgentVector::DeviceAgentVector(CUDAAgent& _cuda_agent, const std::string &_cuda_agent_state, CUDAScatter& _scatter, const unsigned int& _streamId, const cudaStream_t& _stream)
    : AgentVector(_cuda_agent.getAgentDescription(), 0)
    , unbound_buffers_has_changed(false)
    , known_device_buffer_size(_cuda_agent.getStateSize(_cuda_agent_state))
    , cuda_agent(_cuda_agent)
    , cuda_agent_state(_cuda_agent_state)
    , scatter(_scatter)
    , streamId(_streamId)
    , stream(_stream) {
    // Create an empty AgentVector and initialise it manually
    // For each variable create an uninitialised array of variable data
    _size = known_device_buffer_size;
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
    // Grab the unbound variable buffers from the CUDAFatAgentStateList
    // Leave their host counterparts de-allocated until required
    {
        const auto buffs = cuda_agent.getUnboundVariableBuffers(cuda_agent_state);
        for (auto &d_buff : buffs)
            unbound_buffers.emplace_back(d_buff);
    }
}

void DeviceAgentVector::syncChanges() {
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
    for (const auto &ch : change_detail) {
        auto &v = agent->variables.at(ch.first);
        // Copy back variable data into each array
        const char* host_src = static_cast<const char*>(_data->at(ch.first)->getDataPtr());
        char* device_dest = static_cast<char*>(cuda_agent.getStateVariablePtr(cuda_agent_state, ch.first));
        const size_t copy_offset = ch.second.first * v.type_size * v.elements;
        const size_t copy_len = (ch.second.second - ch.second.first) * v.type_size * v.elements;
        gpuErrchk(cudaMemcpyAsync(device_dest + copy_offset, host_src + copy_offset, copy_len, cudaMemcpyHostToDevice, stream));
    }
    change_detail.clear();
    for (const auto& v : agent->variables) {
        // @todo Only copy changed data
    }
    // Copy all unbound buffes
    if (unbound_buffers_has_changed) {
        if (unbound_host_buffer_size != _size) {
            THROW InvalidOperation("Unbound buffers have gone out of sync, in DeviceAgentVector::syncChanges().\n");
        }
        for (auto &buff : unbound_buffers) {
            const size_t variable_size = buff.device->type_size * buff.device->elements;
            gpuErrchk(cudaMemcpyAsync(buff.device->data, buff.host, unbound_host_buffer_size * variable_size, cudaMemcpyHostToDevice, stream));
        }
        unbound_buffers_has_changed = false;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    // Update CUDAAgent statelist size
    cuda_agent.setStateAgentCount(cuda_agent_state, _size);
}

void DeviceAgentVector::initUnboundBuffers() {
    if (!_capacity)
      return;
    const unsigned int device_len = cuda_agent.getStateSize(cuda_agent_state);
    const unsigned int copy_len = _size < device_len ? _size : device_len;
    // Resize to match _capacity
    for (auto &buff : unbound_buffers) {
        if (buff.host) {
            THROW InvalidOperation("Host buffer is already allocated, in DeviceAgentVector::initUnboundBuffers().\n");
        }
        // Alloc
        const size_t var_size = buff.device->type_size * buff.device->elements;
        buff.host = static_cast<char*>(malloc(_capacity * var_size));
        // DtH memcpy
        gpuErrchk(cudaMemcpyAsync(buff.host, buff.device->data, copy_len * var_size, cudaMemcpyDeviceToHost, stream));
        // Not sure this will ever happen, but better safe
        for (unsigned int i = device_len; i < _size; ++i) {
            // We have unknown agents, default init them
            memcpy(buff.host + i * var_size, buff.device->default_value, var_size);
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    unbound_host_buffer_capacity = _capacity;
    unbound_host_buffer_size = copy_len;
    unbound_buffers_has_changed = true;  // Probably not required, but if they are being init, high chance they're going to be changed
}
void DeviceAgentVector::resizeUnboundBuffers(const unsigned int& new_capacity, bool init) {
    // Resize to match agent_count
    for (auto& buff : unbound_buffers) {
        if (!buff.host) {
            THROW InvalidOperation("Not setup to resize before init");
        }
        // Alloc new buff
        const size_t var_size = buff.device->type_size * buff.device->elements;
        char *t = static_cast<char*>(malloc(new_capacity * var_size));
        // Copy data across
        const unsigned int copy_len = _size < unbound_host_buffer_capacity ? _size : unbound_host_buffer_capacity;
        memcpy(t, buff.host, copy_len * var_size);
        // Free old
        free(buff.host);
        // Replace old ptr
        buff.host = t;
        if (init) {
            for (unsigned int i = unbound_host_buffer_capacity; i < new_capacity; ++i) {
                // We have unknown agents, default init them
                memcpy(buff.host + i * var_size, buff.device->default_value, var_size);
            }
        }
    }
    unbound_host_buffer_capacity = new_capacity;
    // unbound_host_buffer_size = agent_count;  // This would only make sense for init, but consisent behaviour is better
    unbound_buffers_has_changed = true;  // Probably not required, but if they are resized, high chance theyre going to change
}

void DeviceAgentVector::_insert(size_type pos, size_type count) {
    // No unbound buffers, return
    if (unbound_buffers.empty() || !count)
        return;
    // Unbound buffers first use, init
    // This updates unbound_host_buffer_size to match known_device_buffer_size
    if (!unbound_host_buffer_capacity)
        initUnboundBuffers();
    // Resizes unbound buffers if necessary
    const size_type new_size = known_device_buffer_size + count;
    if (new_size > unbound_host_buffer_capacity) {
        resizeUnboundBuffers(_capacity, false);
        // Init new agents that won't be init by the replacement below
        for (auto& buff : unbound_buffers) {
            const size_t variable_size = buff.device->type_size * buff.device->elements;
            for (unsigned int i = new_size; i < _capacity; ++i) {
                memcpy(buff.host + i * variable_size, buff.device->default_value, variable_size);
            }
        }
    }
    //  Move all items behind pos, then init all the newly inserted
    for (auto& buff : unbound_buffers) {
        const size_t variable_size = buff.device->type_size * buff.device->elements;
        // Move all items after this index backwards count places
        for (unsigned int i = known_device_buffer_size - 1; i >= pos; --i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(buff.host + (i + count) * variable_size, buff.host + i * variable_size, variable_size);
        }
        // Default init the inserted variables
        for (unsigned int i = pos; i < pos + count; ++i) {
            memcpy(buff.host + i * variable_size, buff.device->default_value, variable_size);
        }
    }
    // Update size
    unbound_buffers_has_changed = true;
    unbound_host_buffer_size = new_size;
    known_device_buffer_size = _size;
    if (unbound_host_buffer_size != _size) {
        THROW InvalidOperation("Unbound buffers have gone out of sync, in DeviceAgentVector::_insert().\n");
    }
    // Update change detail for all variables
    for (const auto& v : agent->variables) {
        // Does it exist in change map
        auto change = change_detail.find(v.first);
        if (change == change_detail.end()) {
            change_detail.emplace(v.first, std::pair<size_type, size_type>{pos, _size});
        } else {
            // Inclusive min bound
            change->second.first = change->second.first > pos ? pos : change->second.first;
            // Exclusive max bound
            change->second.second = _size;
        }
    }
}
void DeviceAgentVector::_erase(size_type pos, size_type count) {
    // No unbound buffers, return
    if (unbound_buffers.empty() || !count)
        return;
    // Unbound buffers first use, init
    if (!unbound_host_buffer_capacity)
        initUnboundBuffers();
    const size_type new_size = known_device_buffer_size - count;
    const size_type copy_start = pos + count;
    for (auto& buff : unbound_buffers) {
        const size_t variable_size = buff.device->type_size * buff.device->elements;
        // Move all items after this index forwards count places
        for (unsigned int i = copy_start; i < unbound_host_buffer_size; ++i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(buff.host + (i - count) * variable_size, buff.host + i * variable_size, variable_size);
        }
        // Default init the empty variables at the end
        for (unsigned int i = new_size; i < known_device_buffer_size; ++i) {
            memcpy(buff.host + i * variable_size, buff.device->default_value, variable_size);
        }
    }
    // Update size
    unbound_buffers_has_changed = true;
    unbound_host_buffer_size = new_size;
    known_device_buffer_size = _size;
    if (unbound_host_buffer_size != _size) {
        THROW InvalidOperation("Unbound buffers have gone out of sync, in DeviceAgentVector::_erase().\n");
    }
    // Update change detail for all variables
    for (const auto &v : agent->variables) {
        // Does it exist in change map
        auto change = change_detail.find(v.first);
        if (change == change_detail.end()) {
            change_detail.emplace(v.first, std::pair<size_type, size_type>{pos, _size});
        } else {
            // Inclusive min bound
            change->second.first = change->second.first > pos ? pos : change->second.first;
            // Exclusive max bound
            change->second.second = _size;
        }
    }
}


void DeviceAgentVector::_changed(const std::string& variable_name, size_type pos) {
    // Check the variable exists
    auto var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable %s was not found, "
            "in DeviceAgentVector::_changed()\n",
            variable_name.c_str());
    }
    // Does it exist in change map
    auto change = change_detail.find(variable_name);
    if (change == change_detail.end()) {
        change_detail.emplace(variable_name, std::pair<size_type, size_type>{pos, pos + 1});
    } else {
        // Inclusive min bound
        change->second.first = change->second.first > pos ? pos : change->second.first;
        // Exclusive max bound
        change->second.second = change->second.second <= pos ? pos + 1 : change->second.second;
    }
}
void DeviceAgentVector::_changedAfter(const std::string& variable_name, size_type pos) {
    // Check the variable exists
    auto var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable %s was not found, "
            "in DeviceAgentVector::_changed()\n",
            variable_name.c_str());
    }
    // Does it exist in change map
    auto change = change_detail.find(variable_name);
    if (change == change_detail.end()) {
        change_detail.emplace(variable_name, std::pair<size_type, size_type>{pos, _size});
    } else {
        // Inclusive min bound
        change->second.first = change->second.first > pos ? pos : change->second.first;
        // Exclusive max bound
        change->second.second = _size;
    }
}
