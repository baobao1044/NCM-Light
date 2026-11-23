/*
    Shared NCM-Light user-configurable shader options.
    Included by both vertex programs (waving, shadow) and fragment/composite
    programs so Iris/OptiFine menu toggles stay synchronized everywhere.
*/

#ifndef NCM_OPTIONS_GLSL
#define NCM_OPTIONS_GLSL

#define FOG_STRENGTH 1.0            // [0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0]
#define RAYS_STRENGTH 1.0           // [0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0]
#define DENSITY_SCALE 1.0           // [0.25 0.5 0.75 1.0 1.5 2.0 3.0]
#define SUN_SCATTER 0.8             // [0.0 0.2 0.35 0.5 0.65 0.8 0.9]
#define RT_GI 1                     // [0 1]
#define PT_SPP 1                    // [1 2 4]
#define PT_MULTI_BOUNCE 1           // [0 1]
#define PT_GUIDING 1                // [0 1]
#define RT_REFLECTIONS 1            // [0 1]
#define RT_STRENGTH 1.0             // [0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0]
#define REFINE_THRESHOLD 0.08       // [0.0 0.02 0.05 0.08 0.12 0.2 0.35 0.5]
#define REFINE_SOFTNESS 0.15        // [0.0 0.05 0.1 0.15 0.25 0.4]
#define DEBUG_LEVEL 0               // [0 1 2 3 4 5 6]
#define SHADOWS 1                   // [0 1]

#define VOLUMETRIC_CLOUDS 1         // [0 1]
#define WAVING_FOLIAGE 1            // [0 1]
#define COLORED_SHADOWS 1           // [0 1]
#define WATER_CAUSTICS 1            // [0 1]
#define SPECTRAL_DISPERSION 1       // [0 1]
#define SUBSURFACE_SCATTER 1        // [0 1]
#define LABPBR_SUPPORT 1            // [0 1]

const int shadowMapResolution = 1024; // [512 1024 2048 4096]
const float shadowDistance = 56.0;
const float SHADOW_DISTORT = 0.78;

#endif
