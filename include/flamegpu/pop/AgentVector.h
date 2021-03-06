#ifndef INCLUDE_FLAMEGPU_POP_AGENTVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_AGENTVECTOR_H_

#include <string>
#include <utility>
#include <memory>
#include <map>

#include "flamegpu/pop/MemoryVector.h"
#include "flamegpu/model/AgentData.h"

class AgentInstance;
class AgentDescription;
class AgentVector_CAgent;
class AgentVector_Agent;

class AgentVector {
    /**
     * Proportion which capacity increases when size must be increased automatically
     * This value must be greater than 1.0
     */
    static const float RESIZE_FACTOR;
    /**
     * CUDAAgentStateList::getAgentData(AgentVector&) uses private AgentVector::resize(size_type, bool)
     * Can't include CUDAAgentStateList to friend the specific method.
     */
    friend class CUDAAgentStateList;

 public:
    typedef unsigned int size_type;
    /**
     * View into the AgentVector to provide mutable access to a specific Agent's data
     */
    typedef AgentVector_Agent Agent;
    /**
     * View into the AgentVector to provide immutable access to a specific Agent's data
     */
    typedef AgentVector_CAgent CAgent;
    typedef std::map<std::string, std::unique_ptr<GenericMemoryVector>> AgentDataMap;

    // member typedefs provided through inheriting from std::iterator
    // They might all be wrong
    class const_iterator;
    class const_reverse_iterator;
    class iterator : public std::iterator<
        std::input_iterator_tag,  // iterator_category
        Agent,                    // value_type
        size_type,                // difference_type
        const Agent*,             // pointer
        Agent                     // reference
    > {
        friend class AgentVector;
        const std::shared_ptr<const AgentData>& _agent;
        const std::weak_ptr<AgentDataMap> _data;
        size_type _pos;
     public:
        operator AgentVector::const_iterator() const {
            return const_iterator(_agent, _data, _pos);
        }
        iterator(const std::shared_ptr<const AgentData>& agent, std::weak_ptr<AgentDataMap> data, size_type pos = 0)
            : _agent(agent), _data(std::move(data)), _pos(pos) { }
        iterator& operator++() { ++_pos; return *this; }
        iterator operator++(int) { iterator retval = *this; ++(*this); return retval; }
        bool operator==(iterator other) const { return _pos == other._pos &&
            (_data.lock() == other._data.lock() || (_data.lock() && other._data.lock() && *_data.lock() == *other._data.lock()));
        }
        bool operator!=(iterator other) const { return !(*this == other); }
        Agent operator*() const;
    };
    class const_iterator : public std::iterator<
        std::input_iterator_tag,  // iterator_category
        CAgent,                   // value_type
        size_type,                // difference_type
        const CAgent*,            // pointer
        CAgent                    // reference
    > {
        friend class AgentVector;
        const std::shared_ptr<const AgentData> &_agent;
        const std::weak_ptr<AgentDataMap> _data;
        size_type _pos;
     public:
        const_iterator(const std::shared_ptr<const AgentData>& agent, std::weak_ptr<AgentDataMap> data, size_type pos = 0)
            : _agent(agent), _data(std::move(data)), _pos(pos) { }
        const_iterator& operator++() { ++_pos; return *this; }
        const_iterator operator++(int) { const_iterator retval = *this; ++(*this); return retval; }
        bool operator==(const_iterator other) const { return _pos == other._pos &&
            (_data.lock() == other._data.lock() || (_data.lock() && other._data.lock() && *_data.lock() == *other._data.lock()));
        }
        bool operator!=(const_iterator other) const { return !(*this == other); }
        CAgent operator*() const;
    };
    class reverse_iterator : public std::iterator<
        std::input_iterator_tag,  // iterator_category
        Agent,                    // value_type
        size_type,                // difference_type
        const Agent*,             // pointer
        Agent                     // reference
    > {
        friend class AgentVector;
        const std::shared_ptr<const AgentData> &_agent;
        const std::weak_ptr<AgentDataMap> _data;
        size_type _pos;
     public:
        operator AgentVector::const_reverse_iterator() const {
            return const_reverse_iterator(_agent, _data, _pos);
        }
        explicit reverse_iterator(const std::shared_ptr<const AgentData>& agent, std::weak_ptr<AgentDataMap> data, size_type pos = 0)
            : _agent(agent), _data(std::move(data)), _pos(pos) { }
        reverse_iterator& operator++() { --_pos; return *this; }
        reverse_iterator operator++(int) { reverse_iterator retval = *this; ++(*this); return retval; }
        bool operator==(reverse_iterator other) const { return _pos == other._pos &&
            (_data.lock() == other._data.lock() || (_data.lock() && other._data.lock() && *_data.lock() == *other._data.lock()));
        }
        bool operator!=(reverse_iterator other) const { return !(*this == other); }
        Agent operator*() const;
    };
    class const_reverse_iterator : public std::iterator<
        std::input_iterator_tag,  // iterator_category
        CAgent,                   // value_type
        size_type,                // difference_type
        const CAgent*,            // pointer
        CAgent                    // reference
    > {
        friend class AgentVector;
        const std::shared_ptr<const AgentData>& _agent;
        const std::weak_ptr<AgentDataMap> _data;
        size_type _pos;
     public:
        explicit const_reverse_iterator(const std::shared_ptr<const AgentData>& agent, std::weak_ptr<AgentDataMap> data, size_type pos = 0)
            : _agent(agent), _data(std::move(data)), _pos(pos) { }
        const_reverse_iterator& operator++() { --_pos; return *this; }
        const_reverse_iterator operator++(int) { const_reverse_iterator retval = *this; ++(*this); return retval; }
        bool operator==(const_reverse_iterator other) const { return _pos == other._pos &&
            (_data.lock() == other._data.lock() || (_data.lock() && other._data.lock() && *_data.lock() == *other._data.lock()));
        }
        bool operator!=(const_reverse_iterator other) const { return !(*this == other); }
        CAgent operator*() const;
    };
    /**
     * Constructs the container with count copies of the agent described by agent_desc
     * initialised with the default values specified by agent_desc.
     * @param agent_desc agent_desc Agent description specifying the agent variables to be represented
     * @param count The size of the container
     */
    explicit AgentVector(const AgentDescription &agent_desc, size_type count = 0);
    /**
     * Copy constructor.
     * Constructs the container with the copy of the contents of other
     * @param other another container to be used as source to initialize the elements of the container with
     */
    AgentVector(const AgentVector &other);
    /**
     * Move constructor
     * Constructs the container with the contents of other using move semantics.
     * other is left in an empty but functional state.
     * @param other another container to be used as source to initialize the elements of the container with
     */
    AgentVector(AgentVector &&other) noexcept;
    /**
     * Copy assignment operator.
     * Replaces the contents with a copy of the contents of other
     */
    AgentVector& operator=(const AgentVector &other);
    /**
     * Move assignment operator.
     * Replaces the contents with those of other using move semantics (i.e. the data in other is moved from other into this container).
     * other is left in an empty but functional state.
     */
    AgentVector& operator=(AgentVector &&other) noexcept;

    // Element access
    /**
     * access specified element with bounds checking
     * @param pos position of the element to return
     * @return Reference to the requested element.
     * @throws std::out_of_range if `!(pos < size())`
     */
    Agent at(size_type pos);
    CAgent at(size_type pos) const;
    /**
     * Returns a reference to the element at specified location pos.
     * @param pos position of the element to return
     * @return Reference to the requested element.
     * @throws std::out_of_range if `!(pos < size())`
     */
    Agent operator[](size_type pos);
    CAgent operator[](size_type pos) const;
    /**
     * Returns a reference to the first element in the container.
     * @return Reference to the first element
     * @throws std::out_of_range if `empty()`
     */
    Agent front();
    CAgent front() const;
    /**
     * Returns a reference to the last element in the container.
     * @return Reference to the last element
     * @throws std::out_of_range if `empty()`
     */
    Agent back();
    CAgent back() const;
    /**
     * Returns pointer to the underlying array serving as element storage for the named variable.
     * @param variable_name Name of the variable array to return
     * @throws InvalidAgentVar Agent does not contain variable variable_name
     * @throws InvalidVarType Agent variable variable_name is not of type T
     * @note Returns nullptr if vector is has not yet allocated buffers.
     */
    template<typename T>
    T *data(const std::string &variable_name);
    template<typename T>
    const T* data(const std::string &variable_name) const;
    void* data(const std::string& variable_name);
    const void* data(const std::string& variable_name) const;

    // Iterators
    iterator begin() noexcept;
    const_iterator begin() const noexcept;
    const_iterator cbegin() const noexcept;
    iterator end() noexcept;
    const_iterator end() const noexcept;
    const_iterator cend() const noexcept;
    reverse_iterator rbegin() noexcept;
    const_reverse_iterator rbegin() const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    reverse_iterator rend() noexcept;
    const_reverse_iterator rend() const noexcept;
    const_reverse_iterator crend() const noexcept;

    // Capacity
    /**
     * Checks if the container has no elements, i.e. whether begin() == end()
     * @return `true` if the container is empty, `false` otherwise
     */
    bool empty() const;
    /**
     * Returns the number of elements in the container, i.e. std::distance(begin(), end())
     * @return The number of elements in the container.
     */
    size_type size() const;
    /**
     * Returns the maximum number of elements the container is able to hold due to system or library implementation limitations,
     * i.e. std::distance(begin(), end()) for the largest container.
     * @return Maximum number of elements.
     * @note This value typically reflects the theoretical limit on the size of the container, at most `std::numeric_limits<difference_type>::max()`.
     * At runtime, the size of the container may be limited to a value smaller than max_size() by the amount of RAM available.
     */
    static size_type max_size();
    /**
     * Increase the capacity of the vector to a value that's greater or equal to new_cap. If new_cap is greater than the current capacity(),
     * new storage is allocated, otherwise the method does nothing.
     *
     * reserve() does not change the size of the vector.
     *
     * If new_cap is greater than capacity(), all iterators, including the past-the-end iterator, and all AgentViews are invalidated.
     * Otherwise, no iterators or references are invalidated.
     * @param new_cap new capacity of the vector
     * @throws std::length_error if new_cap > max_size().
     */
    void reserve(size_type new_cap);
    /**
     * Returns the number of elements that the container has currently allocated space for.
     *
     * @return Capacity of the currently allocated storage.
     */
    size_type capacity() const;
    /**
     * Requests the removal of unused capacity.
     *
     * If reallocation occurs, all iterators, including the past the end iterator, and all references to the elements are invalidated.
     * If no reallocation takes place, no iterators or references are invalidated.
     */
    void shrink_to_fit();

    // Modifiers
    /**
     * Erases all elements from the container. After this call, size() returns zero.
     *
     * Invalidates any references, pointers, or iterators referring to contained elements. Any past-the-end iterators are also invalidated.  
     *
     * Leaves the capacity() of the vector unchanged
     */
    void clear();

    /**
     * Inserts elements at the specified location in the container
     * Inserts value before pos
     *
     * Causes reallocation if the new size() is greater than the old capacity().
     * If the new size() is greater than capacity(), all iterators and references are invalidated.
     * Otherwise, only the iterators and references before the insertion point remain valid.
     * The past-the-end iterator is also invalidated.
     *
     * @throw InvalidAgent If agent type of value does not match
     */
    iterator insert(const_iterator pos, const AgentInstance& value);
    iterator insert(size_type pos, const AgentInstance& value);
    iterator insert(const_iterator pos, const Agent& value);
    iterator insert(size_type pos, const Agent& value);
#ifdef SWIG
    void py_insert(size_type pos, const AgentInstance& value);
    void py_insert(size_type pos, const Agent& value);
#endif
    /**
     * Inserts elements at the specified location in the container
     * Inserts count copies of the value before pos
     *
     * Causes reallocation if the new size() is greater than the old capacity().
     * If the new size() is greater than capacity(), all iterators and references are invalidated.
     * Otherwise, only the iterators and references before the insertion point remain valid.
     * The past-the-end iterator is also invalidated.
     *
     * @throw InvalidAgent If agent type of value does not match
     */
    iterator insert(const_iterator pos, size_type count, const AgentInstance& value);
    iterator insert(size_type pos, size_type count, const AgentInstance& value);
    iterator insert(const_iterator pos, size_type count, const Agent& value);
    iterator insert(size_type pos, size_type count, const Agent& value);
#ifdef SWIG
    void py_insert(size_type pos, size_type count, const AgentInstance& value);
    void py_insert(size_type pos, size_type count, const Agent& value);
#endif
    /**
     * Inserts elements at the specified location in the container
     * Inserts elements from range [first, last) before pos.
     * The behavior is undefined if first and last are iterators into *this
     *
     * Causes reallocation if the new size() is greater than the old capacity().
     * If the new size() is greater than capacity(), all iterators and references are invalidated.
     * Otherwise, only the iterators and references before the insertion point remain valid.
     * The past-the-end iterator is also invalidated.
     *
     * @throw InvalidAgent If agent type of first or last does not match
     */
    template<class InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last);
    template<class InputIt>
    iterator insert(size_type pos, InputIt first, InputIt last);
    /**
     * Erases the specified elements from the container.
     * Removes the element at pos.
     *
     * Invalidates iterators and references at or after the point of the erase,  including the end() iterator.
     *
     * The iterator pos must be valid and dereferenceable.
     * Thus the end() iterator (which is valid, but is not dereferenceable) cannot be used as a value for pos.
     *
     * @param pos iterator to the element to remove
     *
     * @return Iterator following the last removed element
     * @return If pos refers to the last element, then the end() iterator is returned
     * @throw OutOfBoundsException pos >= size()
     */
    iterator erase(const_iterator pos);
    iterator erase(size_type pos);
#ifdef SWIG
    void py_erase(size_type pos);
#endif
    /**
     * Erases the specified elements from the container.
     * Removes the elements in the range [first, last).
     *
     * Invalidates iterators and references at or after the point of the erase,  including the end() iterator.
     *
     * The iterator first does not need to be dereferenceable if first==last: erasing an empty range is a no-op.
     *
     * @param first Iterator to the first item of the range of elements to remove
     * @param last Iterator to after the last item of the range of elements to remove
     *
     * @return Iterator following the last removed element
     * @return if last==end() prior to removal,then the updated end() iterator is returned.
     * @return if [first, last) is an empty range, then last is returned
     * @throw OutOfBoundsException first >= size()
     * @throw OutOfBoundsException last > size()
     */
    iterator erase(const_iterator first, const_iterator last);
    iterator erase(size_type first, size_type last);
#ifdef SWIG
    void py_erase(size_type first, size_type last);
#endif
    /**
     * Appends the given element value to the end of the container.
     * The new element is initialized as a copy of value
     *
     * If the new size() is greater than capacity() then all iterators and references (including the past-the-end iterator) are invalidated.
     * Otherwise only the past-the-end iterator is invalidated.
     *
     * @param value	the value of the element to append
     *
     * @throws InvalidAgent If the agent type of the AgentInstance doesn't match the agent type of the AgentVector
     */
    void push_back(const AgentInstance& value);
    /**
     * Appends a default initialised element to the end of the container
     */
    void push_back();
    /**
     * Removes the last element of the container.
     * Calling pop_back on an empty container results in undefined behavior.
     * Iterators and references to the last element, as well as the end() iterator, are invalidated.
     */
    void pop_back();
    /**
     * Resizes the container to contain count elements.
     *
     * If the current size is greater than count, the container is reduced to its first count elements.
     *
     * If the current size is less than count, additional default agents are appended
     * @param count size of the container
     */
    void resize(size_type count);
    /**
     * Exchanges the contents of the container with those of other. Does not invoke any move, copy, or swap operations on individual elements.
     * All iterators and references remain valid. The past-the-end iterator is invalidated.
     */
    void swap(AgentVector& other) noexcept;
    /**
     * Checks if the contents of lhs and rhs are equal,
     * that is, they have the same number of elements and each element in lhs compares equal with the element in rhs at the same position.
     */
    bool operator==(const AgentVector &other) const;
    bool operator!=(const AgentVector &other) const;

    // Util
    /**
     * Returns the agent name from the internal agent description
     */
    std::string getAgentName() const { return agent->name; }
    /**
     * Returns true, if the provided agent description matches the internal agent description of the vector
     */
    bool matchesAgentType(const AgentData &other) const;
    bool matchesAgentType(const AgentDescription& other) const;
    /**
     * Returns the type_index of the named variable
     * @throw InvalidAgentVar When variable_name is not valid
     */
    std::type_index getVariableType(const std::string& variable_name) const;
    /**
     * Returns the full map of variable metadata from the internal agent description
     */
    const VariableMap &getVariableMetaData() const;
    /**
     * Returns the initial state of the internal agent description
     */
    std::string getInitialState() const;

 private:
    /**
     * Resizes the internal vector
     * Note, this version only updates _capacity, _size remains unchanged.
     */
    void resize(size_type count, bool init);
    std::shared_ptr<const AgentData> agent;
    size_type _size;
    size_type _capacity;
    std::shared_ptr<AgentDataMap> _data;
};

#include "flamegpu/pop/AgentVector_Agent.h"

template<typename T>
T* AgentVector::data(const std::string& variable_name) {
    // Is variable name found
    const auto &var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent '%s', "
            "in AgentVector::data().",
            variable_name.c_str(), agent->name.c_str());
    }
    if (std::type_index(typeid(T)) != var->second.type) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector::data().",
            variable_name.c_str(), var->second.type.name(), typeid(T).name());
    }
    // Does the map have a vector
    const auto& map_it = _data->find(variable_name);
    if (map_it != _data->end())
        return static_cast<T*>(map_it->second->getDataPtr());
    return nullptr;
}
template<typename T>
const T* AgentVector::data(const std::string& variable_name) const {
    // Is variable name found
    const auto& var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent '%s', "
            "in AgentVector::data().",
            variable_name.c_str(), agent->name.c_str());
    }
    if (std::type_index(typeid(T)) != var->second.type) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector::data().",
            variable_name.c_str(), var->second.type.name(), typeid(T).name());
    }
    // Does the map have a vector
    const auto& map_it = _data->find(variable_name);
    if (map_it != _data->end())
        return static_cast<T*>(map_it->second->getDataPtr());
    return nullptr;
}

template<class InputIt>
AgentVector::iterator AgentVector::insert(const_iterator pos, InputIt first, InputIt last) {
    if (pos._agent != agent && *pos._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided to pos, '%' required, "
            "in AgentVector::push_back().\n",
            last._agent->name.c_str(), agent->name.c_str());
    }
    return insert(pos._pos, first, last);
}

template<class InputIt>
AgentVector::iterator AgentVector::insert(size_type pos, InputIt first, InputIt last) {
    // Insert elements inrange first-last before pos
    if (first == last)
        return iterator(agent, _data, pos);
    // Confirm they are for the same agent type
    if (first._agent != agent && *first._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided to first, '%' required, "
            "in AgentVector::push_back().\n",
            first._agent->name.c_str(), agent->name.c_str());
    }
    if (last._agent != agent && *last._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided to last, '%' required, "
            "in AgentVector::push_back().\n",
            last._agent->name.c_str(), agent->name.c_str());
    }
    // Expand capacity if required
    const size_type first_copy_index = first._pos < last._pos ? first._pos : last._pos;
    const size_type end_copy_index = first._pos < last._pos ? last._pos : first._pos;
    const size_type copy_count = end_copy_index - first_copy_index;
    {
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + copy_count > new_capacity) {
            new_capacity = static_cast<size_type>(new_capacity * RESIZE_FACTOR) + 1;
        }
        resize(new_capacity, true);
    }
    // Get first index;
    const size_type insert_index = pos;
    // Fix each variable
    auto first_data = first._data.lock();
    if (!first_data) {
        THROW ExpiredWeakPtr("The AgentVector which owns the passed iterators has been deallocated, "
            "in AgentVector::insert().\n");
    }
    for (const auto& v : agent->variables) {
        const auto it = _data->find(v.first);
        char* t_data = static_cast<char*>(it->second->getDataPtr());
        const size_t variable_size = v.second.type_size * v.second.elements;
        // Move all items after this index backwards count places
        for (unsigned int i = _size - 1; i >= insert_index; --i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(t_data + (i + copy_count) * variable_size, t_data + i * variable_size, variable_size);
        }
        // Copy across item data
        const auto other_it = first_data->find(v.first);
        const char* o_data = static_cast<const char*>(other_it->second->getReadOnlyDataPtr());
        memcpy(t_data + insert_index * variable_size, o_data + first_copy_index * variable_size, copy_count * variable_size);
    }
    // Increase size
    _size += copy_count;
    // Return iterator to first inserted item
    return iterator(agent, _data, insert_index);
}

#ifdef SWIG
void AgentVector::py_insert(size_type pos, const AgentInstance& value) {
    insert(pos, value);
}
void AgentVector::py_insert(size_type pos, const Agent& value) {
    insert(pos, value);
}
void AgentVector::py_insert(size_type pos, size_type count, const AgentInstance& value) {
    insert(pos, count, value);
}
void AgentVector::py_insert(size_type pos, size_type count, const Agent& value) {
    insert(pos, count, value);
}
void AgentVector::py_erase(size_type pos) {
    erase(pos);
}
void AgentVector::py_erase(size_type first, size_type last) {
    erase(first, last);
}
#endif
#endif  // INCLUDE_FLAMEGPU_POP_AGENTVECTOR_H_
