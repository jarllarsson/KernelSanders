#ifndef RAYTRACE_SETUP_H
#define RAYTRACE_SETUP_H

// =======================================================================================
//                                      RaytraceSetup
// =======================================================================================

#define MAXSPHERES 9
#define MAXPLANES 1
#define MAXTRIS 100
#define MAXBOXES 4
#define MAXLIGHTS 10
#define MAXMESHLOCAL_VERTSBIN 120
#define MAXMESHLOCAL_INDICESBIN 120 // estimate

#define SPACEDIST 200.0
#define SPHERERADIUS 5.0

#define PI 3.141592653589793238462643383279502884197169399375105820
#define TWOPI 2.0*PI
#define TORAD PI/180
#define TODEG 180/PI
#define PIOVER180 TORAD
#define UCHAR_COLOR_TO_FLOAT 0.00390625f

#define MAX_INTERSECT_DIST 9999999.0f



#endif