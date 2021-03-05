#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_HSVINTERPOLATION_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_HSVINTERPOLATION_H_

#include <string>

#include "flamegpu/visualiser/color/ColorFunction.h"

class HSVInterpolation : public ColorFunction {
public:
    /**
     * 0 = Red, 1 = Green
     * @param variable_name float agent variable to map to the color
     */
    static HSVInterpolation REDGREEN(const std::string &variable_name);
    /**
     * 0 = Green, 1 = Red
     * @param variable_name float agent variable to map to the color
     */
    static HSVInterpolation GREENRED(const std::string& variable_name);
    /**
     * Constructs a static color
     * All components must be provided in the inclusive range [0.0, 1.0]
     * @param variable_name Name of the agent variable which maps to hue, the variable type must be float
     * @param hMin Hue value when the agent variable is 0.0
     * @param hMax Hue value when the agent variable is 1.0
     * @param s Saturation (the inverse amount of grey)
     * @param v Value (brightness)
     */
    HSVInterpolation(const std::string& variable_name, const float& hMin, const float& hMax, const float& s = 1.0f, const float& v = 0.88f);
    /**
     * Set the bounds to clamp an agent variable to before using for HSV interpolation
     * @param min_bound The agent variable value that should map to the minimum hue, must be smaller than max_bound
     * @param max_bound The agent variable value that should map to the maximum hue, must be larger than min_bound
     * @return Returns itself, so that you can chain the method (otherwise constructor would have too many optional args)
     * @throws InvalidArgument if min_bound > max_bound
     */
    HSVInterpolation &setBounds(const float& min_bound, const float& max_bound);
    /**
     * Returns a function returning a constant color in the form:
     * vec4 calculateColor() {
     *   return vec4(1.0, 0.0, 0.0, 1.0);
     * }
     */
    std::string getSrc() const override;
    /**
     * Always returns "color_arg"
     */
    std::string getSamplerName() const override;
    /**
     * Always returns "color_arg"
     */
    std::string getAgentVariableName() const override;

private:
    /**
     * Bounds that the agent variable is clamped to
     * Defaults to [0.0, 1.0]
     */
    float min_bound, max_bound;
    /**
     * Hue must be in the inclusive range [0.0, 360.0] 
     */
    const float hue_min, hue_max;
    /**
     * Sat must be in the inclusive range [0.0, 1.0]
     */
    const float saturation;
    /**
     * Val must be in the inclusive range [0.0, 1.0]
     */
    const float val;
    /**
     * Value returned by getAgentVariableName()
     */
    const std::string variable_name;
};

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_HSVINTERPOLATION_H_
