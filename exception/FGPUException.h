/**
* \brief     FGPUException.cpp
* \details   Provides consistent interface to handle errors through the throw expression. All exceptions generated by the standard library inherit from FGPUException.
* \authors
* \note      The Javadoc style was used. The JAVADOC_AUTOBRIEF should be set to yes
* \see       http://fnch.users.sourceforge.net/doxygen_c.html  for commenting style
* \version
* \date      Dec 2016
* \bug
* \warning
* \copyright
* Created on: 09 Dec 2016
*/

#ifndef _FGPUEXCEPTION
#define _FGPUEXCEPTION

#include <string>
#include <iostream>
#include <exception>

using namespace std;

/*! Class for unknown exceptions thrown*/
class UnknownError {};


/*! Base class for exceptions thrown */
class FGPUException//: public exception
{

public:
    /**
     * A constructor
     * @brief Constructs the FGPUException object
     */
    FGPUException(const char * msg="Unknown error msg")
    {
        err_message = msg;
    };

    /**
     * A destructor.
     * @brief Destroys the FGPUException object
     */
    ~FGPUException() {};

    /**
    * @brief Returns the explanatory string
    * @param none
    * @return Pointer to a null-terminated string with explanatory information. The pointer is guaranteed to be valid at least until the exception object from which it is obtained is destroyed, or until a non-const member function on the FGPUException object is called.
    */
    virtual const char *what() const
    {
        return err_message;
    }
protected:
    const char * err_message;
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid input file. Situations where the input file does not exist or cannot be read by the program.
*/
class InvalidInputFile: public FGPUException
{
public:

    InvalidInputFile(const char * msg="Invalid Input File"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It is used to report errors when hash list is full.
*/
class InvalidHashList : public FGPUException
{
public:
    InvalidHashList(const char *msg="Hash list full. This should never happen"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid agent variable type. This could happen when retriving or setting a variable of differet type.
*/
class InvalidVarType : public FGPUException
{
public:
    InvalidVarType(const char *msg="Bad variable type in agent instance set/get variable"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid agent state name.
*/
class InvalidStateName : public FGPUException
{
public:
    InvalidStateName(const char *msg= "Invalid agent state name"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid map entry.
*/
class InvalidMapEntry : public FGPUException
{
public:
    InvalidMapEntry(const char *msg="Missing entry in type sizes map. Something went bad."):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid agent memory variable type.
*/
class InvalidAgentVar : public FGPUException
{
public:
    InvalidAgentVar(const char *msg="Invalid agent memory variable"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};


/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid CUDA agent variable.
*/
class InvalidCudaAgent: public FGPUException
{
public:
    InvalidCudaAgent(const char *msg="CUDA agent not found. This should not happen"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid CUDA agent map size (i.e.map size is qual to zero).
*/
class InvalidCudaAgentMapSize : public FGPUException
{
public:
    InvalidCudaAgentMapSize(const char *msg="CUDA agent map size is zero"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid CUDA agent description.
*/
class InvalidCudaAgentDesc : public FGPUException
{
public:
    InvalidCudaAgentDesc(const char *msg="CUDA Agent uses different agent description"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid agent variable type. This could happen when retriving or setting a variable of differet type.
*/
class InvalidAgentFunc : public FGPUException
{
public:
    InvalidAgentFunc(const char *msg="Unknown agent function"):FGPUException(msg) {}
    /**
    * @brief Returns the explanatory string
    * @see FGPUException.what()
    */
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid function layer index.
*/
class InvalidFuncLayerIndx : public FGPUException
{
public:
    InvalidFuncLayerIndx(const char *msg= "Agent function layer index out of bounds!"):FGPUException(msg) {}
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid population data.
*/
class InvalidPopulationData : public FGPUException
{
public:
    InvalidPopulationData(const char *msg= "Invalid Population data"):FGPUException(msg) {}
    virtual const char *what() const
    {
        return err_message;
    }
};

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to invalid memory capacity.
*/
class InvalidMemoryCapacity : public FGPUException
{
public:
    InvalidMemoryCapacity(const char *msg= "Invalid Memory Capacity"):FGPUException(msg) {}
    virtual const char *what() const
    {
        return err_message;
    }
};

#endif //_FGPUEXCEPTION
