/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 Module:       FGWinds.cpp
 Author:       Jon Berndt, Tony Peden, Andreas Gaeb
 Date started: Extracted from FGAtmosphere, which originated in 1998
               5/2011
 Purpose:      Models winds, gusts, turbulence, and other atmospheric disturbances
 Called by:    FGFDMExec

 ------------- Copyright (C) 2011  Jon S. Berndt (jon@jsbsim.org) -------------

 This program is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License as published by the Free Software
 Foundation; either version 2 of the License, or (at your option) any later
 version.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License along with
 this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 Place - Suite 330, Boston, MA  02111-1307, USA.

 Further information about the GNU Lesser General Public License can also be found on
 the world wide web at http://www.gnu.org.

FUNCTIONAL DESCRIPTION
--------------------------------------------------------------------------------

HISTORY
--------------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
COMMENTS, REFERENCES,  and NOTES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[1]   Anderson, John D. "Introduction to Flight, Third Edition", McGraw-Hill,
      1989, ISBN 0-07-001641-0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INCLUDES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>
#include </localdata/giersch/netcdf/netcdf-cxx-v4.3.0/include/netcdf>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include "FGWinds.h"
#include "FGFDMExec.h"

#include <Main/fg_props.hxx>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

namespace JSBSim {

IDENT(IdSrc,"$Id: FGWinds.cpp,v 1.15 2015/02/27 20:49:36 bcoconni Exp $");
IDENT(IdHdr,ID_WINDS);

// Return this in event of a problem.
static const int NC_ERR = 2;

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CLASS IMPLEMENTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// square a value, but preserve the original sign

/*
static inline double square_signed (double value)
{
    if (value < 0)
        return value * value * -1;
    else
        return value * value;
}
*/

/// simply square a value
static inline double sqr(double x) { return x*x; }

FGWinds::FGWinds(FGFDMExec* fdmex) : FGModel(fdmex)
{
  Name = "FGWinds";

  MagnitudedAccelDt = MagnitudeAccel = Magnitude = TurbDirection = 0.0;
  SetTurbType( ttMilspec );
  TurbGain = 1.0;
  TurbRate = 10.0;
  Rhythmicity = 0.1;
  spike = target_time = strength = 0.0;
  wind_from_clockwise = 0.0;
  psiw = 0.0;

  vGustNED.InitMatrix();
  vTurbulenceNED.InitMatrix();
  vCosineGust.InitMatrix();

  // Milspec turbulence model
  windspeed_at_20ft = 0.;
  probability_of_exceedence_index = 0;
  POE_Table = new FGTable(7,12);
  // this is Figure 7 from p. 49 of MIL-F-8785C
  // rows: probability of exceedance curve index, cols: altitude in ft
  *POE_Table
           << 500.0 << 1750.0 << 3750.0 << 7500.0 << 15000.0 << 25000.0 << 35000.0 << 45000.0 << 55000.0 << 65000.0 << 75000.0 << 80000.0
    << 1   <<   3.2 <<    2.2 <<    1.5 <<    0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0
    << 2   <<   4.2 <<    3.6 <<    3.3 <<    1.6 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0
    << 3   <<   6.6 <<    6.9 <<    7.4 <<    6.7 <<     4.6 <<     2.7 <<     0.4 <<     0.0 <<     0.0 <<     0.0 <<     0.0 <<     0.0
    << 4   <<   8.6 <<    9.6 <<   10.6 <<   10.1 <<     8.0 <<     6.6 <<     5.0 <<     4.2 <<     2.7 <<     0.0 <<     0.0 <<     0.0
    << 5   <<  11.8 <<   13.0 <<   16.0 <<   15.1 <<    11.6 <<     9.7 <<     8.1 <<     8.2 <<     7.9 <<     4.9 <<     3.2 <<     2.1
    << 6   <<  15.6 <<   17.6 <<   23.0 <<   23.6 <<    22.1 <<    20.0 <<    16.0 <<    15.1 <<    12.1 <<     7.9 <<     6.2 <<     5.1
    << 7   <<  18.7 <<   21.5 <<   28.4 <<   30.2 <<    30.7 <<    31.0 <<    25.2 <<    23.1 <<    17.5 <<    10.7 <<     8.4 <<     7.2;

  bind();
  Debug(0);
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FGWinds::~FGWinds()
{
  delete(POE_Table);
  Debug(1);
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool FGWinds::InitModel(void)
{
  if (!FGModel::InitModel()) return false;

  psiw = 0.0;

  vGustNED.InitMatrix();
  vTurbulenceNED.InitMatrix();
  vCosineGust.InitMatrix();

  oneMinusCosineGust.gustProfile.Running = false;
  oneMinusCosineGust.gustProfile.elapsedTime = 0.0;

  return true;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool FGWinds::Run(bool Holding)
{
  if (FGModel::Run(Holding)) return true;
  if (Holding) return false;

//  if (turbType != ttNone) Turbulence(in.AltitudeASL);
  if (turbType != ttNone) Turbulence(in.DistanceAGL); //Use height above ground and not above mean sea level (AltitudeASL) for the turbulence calculation
  if (oneMinusCosineGust.gustProfile.Running) CosineGust();
 
  vTotalWindNED = vWindNED + vGustNED + vCosineGust + vTurbulenceNED;
  
  //Get wind in the body frame
  vWindBody       = in.Tl2b * vWindNED;  
  vTurbulenceBody = in.Tl2b * vTurbulenceNED;
  vTotalWindBody  = in.Tl2b * vTotalWindNED;
  
   // psiw (Wind heading) is the direction the wind is blowing towards
  if (vWindNED(eX) != 0.0) psiw = atan2( vWindNED(eY), vWindNED(eX) );
  if (psiw < 0) psiw += 2*M_PI;

  Debug(2);
  return false;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// psi is the angle that the wind is blowing *towards*

void FGWinds::SetWindspeed(double speed)
{
  if (vWindNED.Magnitude() == 0.0) {
    psiw = 0.0;
    vWindNED(eNorth) = speed;
  } else {
    vWindNED(eNorth) = speed * cos(psiw);
    vWindNED(eEast) = speed * sin(psiw);
    vWindNED(eDown) = 0.0;
  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

double FGWinds::GetWindspeed(void) const
{
  return vWindNED.Magnitude();
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// psi is the angle that the wind is blowing *towards*

void FGWinds::SetWindPsi(double dir)
{
  double mag = GetWindspeed();
  psiw = dir;
  SetWindspeed(mag);
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int FGWinds::InputPALM()
{
  try
  {
  cout << "*** Input and initialization of PALM wind data" << endl;

  // Open the file for read access
  //NcFile dataFile("/localdata/giersch/Masterarbeit/PALM_Winddaten/Assist_gauss_6_3d_t_end/Assist_gauss_6_3d_t_end_netcdf4.nc", NcFile::read);
  
  //NcFile dataFile("/localdata/giersch/PALM_OUTPUT/KSFO_homogen_270deg/OUTPUT/KSFO_homogen_270deg_30ms_m01.2.nc", NcFile::read);
  //NcFile dataFile2("/localdata/giersch/PALM_OUTPUT/KSFO_homogen_270deg/OUTPUT/KSFO_homogen_270deg_30ms_m01_av.2.nc", NcFile::read);
  
  NcFile dataFile("/localdata/giersch/PALM_OUTPUT/KSFO_heterogeneous_291deg/OUTPUT/KSFO_heterogeneous_291deg_m01.1.nc", NcFile::read);
  NcFile dataFile2("/localdata/giersch/PALM_OUTPUT/KSFO_heterogeneous_291deg/OUTPUT/KSFO_heterogeneous_291deg_m01_av.1.nc", NcFile::read);

  //NcFile dataFile("/localdata/giersch/Masterarbeit/PALM_Winddaten/1-cos_gust.nc", NcFile::read);
  
  // Essential for determining the dimension. Calling of getSize becomes possible
  NcDim dim;

  // Retrieve the variable named "u", "v", "w", ...
  // Instantaneous velocity data
  NcVar data_u=dataFile.getVar("u");
  if(data_u.isNull()) return NC_ERR;

  NcVar data_v=dataFile.getVar("v");
  if(data_v.isNull()) return NC_ERR;

  NcVar data_w=dataFile.getVar("w");
  if(data_w.isNull()) return NC_ERR;

  // Retrieve the viable named "x", "y", and "zw_3d"
  NcVar data_x=dataFile.getVar("x");
  if(data_x.isNull()) return NC_ERR;

  NcVar data_y=dataFile.getVar("y");
  if(data_y.isNull()) return NC_ERR;

  NcVar data_z=dataFile.getVar("zw_3d");
  if(data_z.isNull()) return NC_ERR;

  // Retrieve the dimension of the z-axis
  dim = data_u.getDim(0); 
  
  size_t dimt = dim.getSize();
  
  // Retrieve the dimension of the z-axis
  dim = data_u.getDim(1);
  //dim = data_u.getDim(0);
  size_t dimz = dim.getSize();

  // Retrieve the dimension of the y-axis
  dim = data_u.getDim(2);
  //dim = data_u.getDim(1);
  size_t dimy = dim.getSize();

  // Retrieve the dimension of the x-axis
  dim = data_u.getDim(3);
  //dim = data_u.getDim(2);
  size_t dimx = dim.getSize();

  // Declare PALM wind and spatial components
  float x_palm[NX];
  float y_palm[NY];
  float z_palm[NZ];

  // Declare start Vector specifying the index in the variable where the first
  // of the data values will be read.
  std::vector<size_t> start(4);

  start[0] = dimt - 1;
  start[1] = 0;
  start[2] = 0;
  start[3] = 0;

  // Declare count Vector specifying the edge lengths along each dimension of
  // the block of data values to be read.
  std::vector<size_t> count(4);
 
  // count[0] = NZ
  // count[1] = NY
  // count[2] = NX
  count[0] = 1;
  count[1] = NZ;
  count[2] = NY;
  count[3] = NX;

  // Write data_* to the *_palm fields
  data_u.getVar(start,count,u_palm);
  data_v.getVar(start,count,v_palm);
  data_w.getVar(start,count,w_palm);
  data_x.getVar(x_palm);
  data_y.getVar(y_palm);
  data_z.getVar(z_palm);

  // Calculate grid spacing
  delta_x = x_palm[1]- x_palm[0];
  delta_y = y_palm[1]- y_palm[0];
  delta_z = z_palm[1]- z_palm[0];

  // Determine model size
  xsize = (dimx-1)*delta_x;
  ysize = (dimy-1)*delta_y;
  zsize = (dimz-1)*delta_z;

  //Inertial airplane position in PALM wind field [m] (center of the model domain in homogeneous case and in line with the centerline of the runway
  //and downstream the idealized building in the hereogeneous case)
  if (windmodel == "PALM3D_homogeneous"){
      xp_old = xsize / 2.0; // x is defined in west-east direction
      yp_old = ysize / 2.0; // y is defined in south-north direction
  }
  else if (windmodel == "PALM3D_heterogeneous"){
      xp_old = 2765.0; // x is defined in west-east direction
      yp_old = 0.0;    // y is defined in south-north direction    
  }
  zp_old = in.DistanceAGL * FT2M; // position/altitude-ft = Höhe über NN
  distance_h = 0.0; // covered distance (only horizontal)
  
  Setnode = fgGetNode("/local-weather/PALM/x-position",true);
  Setnode->setFloatValue(xp_old);
  
  Setnode = fgGetNode("/local-weather/PALM/y-position",true);
  Setnode->setFloatValue(yp_old);
  
  Setnode = fgGetNode("/local-weather/PALM/z-position",true);
  Setnode->setFloatValue(zp_old);
  
  // Additional data and rotated data is needed for the heterogeneous case
  if (windmodel == "PALM3D_heterogeneous"){
     // Retrieve the variable named "u", "v", "w"
     // Time averaged velocity data
     NcVar data_u_av=dataFile2.getVar("u");
     if(data_u_av.isNull()) return NC_ERR;

     NcVar data_v_av=dataFile2.getVar("v");
     if(data_v_av.isNull()) return NC_ERR;

     NcVar data_w_av=dataFile2.getVar("w");
     if(data_w_av.isNull()) return NC_ERR;

     // Write data_* to the *_palm fields  
     data_u_av.getVar(start,count,u_av_palm);
     data_v_av.getVar(start,count,v_av_palm);
     data_w_av.getVar(start,count,w_av_palm);

     // Rotate wind values according to rotationangle
     for(int k = 0; k < dimz; k++)
     {
        for(int j = 0; j < dimy; j++)
        {
           for(int i = 0; i < dimx; i++) 
           {
              u_palm_rotated[k][j][i] = cos(rotationangle * M_PI/180.0) * u_palm[k][j][i] - sin(rotationangle * M_PI/180.0) * v_palm[k][j][i]; 
              v_palm_rotated[k][j][i] = sin(rotationangle * M_PI/180.0) * u_palm[k][j][i] + cos(rotationangle * M_PI/180.0) * v_palm[k][j][i]; 
              u_av_palm_rotated[k][j][i] = cos(rotationangle * M_PI/180.0) * u_av_palm[k][j][i] - sin(rotationangle * M_PI/180.0) * v_av_palm[k][j][i]; 
              v_av_palm_rotated[k][j][i] = sin(rotationangle * M_PI/180.0) * u_av_palm[k][j][i] + cos(rotationangle * M_PI/180.0) * v_av_palm[k][j][i];
              
              u_palm[k][j][i]    = u_palm_rotated[k][j][i]; 
              v_palm[k][j][i]    = v_palm_rotated[k][j][i];
              u_av_palm[k][j][i] = u_av_palm_rotated[k][j][i]; 
              v_av_palm[k][j][i] = v_av_palm_rotated[k][j][i]; 
           }
        }
     }
     
  }

  input_PALM_data_flag = 1;
  Setnode = fgGetNode("local-weather/PALM/input-data-flag",false);
  Setnode->setBoolValue(input_PALM_data_flag);
  

  // The netCDF file is automatically closed by the NcFile destructor
  cout << "*** Input and initialization of PALM wind data successfully" << endl;
  

  cout << "*** Determine wind data for the initial position" << endl;

  // Determine velocity components for the initial position (tri-linear interpolation)
  // First: u-component at C.G.

  // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old >= xsize){
      xp_old -= xsize;
  }

  if (xp_old < 0.0){
      xp_old += xsize;
  }

  // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old >= ysize + delta_y / 2.0){
      yp_old -= ysize;
  }

  if (yp_old < 0.0 + delta_y / 2.0){
      yp_old += ysize;
  }

  // Query if zp_old leaves the boundary
  if (zp_old >= zsize - delta_z/2.0){
      zp_old = zsize - (delta_z/2.0 + 0.000001);
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old / delta_x);       // Relevant index for x-direction
  j = int(yp_old / delta_y - 0.5); // Relevant index for y-direction
  k = int(zp_old / delta_z + 0.5); // Relevant index for z-direction

  // Determie u at aircraft position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
     u_windspeed = help3 + (help6 - help3) *
     (zp_old / (0.5 * delta_z)); // u_value at x= xp_old and y= yp_old and z=zp_old
  }
  else
     u_windspeed = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // u_value at x= xp_old and y= yp_old and z=zp_old

     
     

  // Second: v-component at C.G.

  // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old >= xsize + delta_x / 2.0){
      xp_old -= xsize;
  }

  if (xp_old < 0.0 + delta_x / 2.0){
      xp_old += xsize;
  }

  // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old >= ysize){
      yp_old -= ysize;
  }

  if (yp_old < 0.0){
      yp_old += ysize;
  }

  // Query if zp_old leaves the boundary
  if (zp_old >= zsize - delta_z/2.0){
      zp_old = zsize - (delta_z/2.0 + 0.000001);
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_old / delta_y);       // Relevant index for y-direction

  // Determie v at aircraft position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
     v_windspeed = help3 + (help6 - help3) *
     (zp_old / (0.5 * delta_z)); // v_value at x= xp_old and y= yp_old and z=zp_old
  }
  else {
     v_windspeed = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // v_value at x= xp_old and y= yp_old and z=zp_old
  }

  
  

  // Third: w-component at C.G.

  // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old >= xsize + delta_x / 2.0){
      xp_old -= xsize;
  }

  if (xp_old < 0.0 + delta_x / 2.0){
      xp_old += xsize;
  }

  // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old >= ysize + delta_y / 2.0){
      yp_old -= ysize;
  }

  if (yp_old < 0.0 + delta_y / 2.0){
      yp_old += ysize;
  }

  // Query if zp_old leaves the boundary
  if (zp_old > zsize){
      zp_old = zsize - 0.000001;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_old / delta_y - 0.5); // Relevant index for y-direction
  k = int(zp_old / delta_z);       // Relevant index for z-direction

  // Determie w at aircraft position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_old / delta_x - i - 0.5); // w_value at x=xp_old and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_old / delta_x - i - 0.5); // w_value at xp_old and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_old / delta_y) - j - 0.5); // w_value at x= xp_old and y= yp_old and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_old / delta_x - i - 0.5); //w_value at x=xp_old and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_old / delta_x - i - 0.5); // w_value at x=xp_old and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_old / delta_y) - j- 0.5); // w_value at x= xp_old and y= yp_old and z=(k+1)*delta_z

  w_windspeed = help3 + (help6 - help3) *
  ((zp_old / delta_z) - k); // w_Wert at x= xp_old and y= yp_old and z=zp_old

  
  

  // Fourth: wind at wing tips for p
  // indices _1,_2,_3 stands for the positions described in the 4 point model of
  // B. Etkin: "Turbulent wind and its effect on flight" (1981).
  
  // Get the heading of the aircraft
  Getnode = fgGetNode("orientation/heading-deg",false);
  phi_heading = Getnode->getFloatValue(); // North direction = 0, East direction = 90 ....

  // Determine position of the right and left wing tip ((Assumption:
  // Aircraft roll angle is 0. The wings are aligned along the horinzontal plane and have
  // no vertical displacement.)
  alpha = -(phi_heading - 360) * (M_PI/180.0);
  dx = cos(alpha) * in.wingspan/2.0 * FT2M;  
  dy = sin(alpha) * in.wingspan/2.0 * FT2M;  
  
  // w at right wing
  float xp_old_1 = xp_old + dx;
  float yp_old_1 = yp_old + dy;
  
  // Query if xp_old_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_1 >= xsize + delta_x / 2.0){
      xp_old_1 -= xsize;
  }

  if (xp_old_1 < 0.0 + delta_x / 2.0){
      xp_old_1 += xsize;
  }
  
  // Query if yp_old_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_1 >= ysize + delta_y / 2.0){
      yp_old_1 -= ysize;
  }

  if (yp_old_1 < 0.0 + delta_y / 2.0){
      yp_old_1 += ysize;
  }
  
  // Determine relevant indices next to xp_old_1,yp_old_1, int()
  // reduced the input value to the next lower integer value
  // relevant indices j and k stay equal from the previous calculation for w
  i = int(xp_old_1 / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_old_1 / delta_y - 0.5); // Relevant index for y-direction
  
  // Determie w at xp_old_1, yp_old_1, z_old position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_old_1 / delta_x - i - 0.5); // w_value at x=xp_old_1 and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_old_1 / delta_x - i - 0.5); // w_value at xp_old_1 and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_old_1 / delta_y) - j - 0.5); // w_value at x= xp_old_1 and y= yp_old_1 and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_old_1 / delta_x - i - 0.5); // w_value at x=xp_old_1 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_old_1 / delta_x - i - 0.5); // w_value at x=xp_old_1 and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_old_1 / delta_y) - j- 0.5); // w_value at x= xp_old_1 and y= yp_old_1 and z=(k+1)*delta_z

  w_windspeed_1 = help3 + (help6 - help3) *
  ((zp_old / delta_z) - k); // w_Wert at x= xp_old_1 and y= yp_old and z=zp_old

  
  
  
  // w at left wing
  float xp_old_2 = xp_old - dx;
  float yp_old_2 = yp_old - dy;
  
  // Query if xp_old_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_2 >= xsize + delta_x / 2.0){
      xp_old_2 -= xsize;
  }

  if (xp_old_2 < 0.0 + delta_x / 2.0){
      xp_old_2 += xsize;
  }
  
  // Query if yp_old_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_2 >= ysize + delta_y / 2.0){
      yp_old_2 -= ysize;
  }

  if (yp_old_2 < 0.0 + delta_y / 2.0){
      yp_old_2 += ysize;
  }
  
  // Determine relevant indices next to xp_old_2,yp_old_2, int()
  // reduced the input value to the next lower integer value
  // relevant indices j and k stay equal from the previous calculation for w
  i = int(xp_old_2 / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_old_2 / delta_y - 0.5); // Relevant index for y-direction

  // Determie w at xp_old_2, yp_old_2, z_old position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_old_2 / delta_x - i - 0.5); // w_value at x=xp_old_2 and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_old_2 / delta_x - i - 0.5); //w_value at xp_old_2 and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_old_2 / delta_y) - j - 0.5); // w_value at x= xp_old_2 and y= yp_old_2 and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_old_2 / delta_x - i - 0.5); // w_value at x=xp_old_2 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_old_2 / delta_x - i - 0.5); // w_value at x=xp_old_2 and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_old_2 / delta_y) - j- 0.5); // w_value at x= xp_old_2 and y= yp_old_2 and z=(k+1)*delta_z

  w_windspeed_2 = help3 + (help6 - help3) *
  ((zp_old / delta_z) - k); // w_Wert at x= xp_old_2 and y= yp_old_2 and z=zp_old

  
  

  // Fifth: wind at tail fin for q-component
  // Determine position of the fin ((Assumption: Aircraft pitch angle is 0. 
  // The longitudinal axis is aligned along the horinzontal plane and have
  // no vertical displacement.)
  dx = sin(alpha) * vtailarm;  
  dy = cos(alpha) * vtailarm;  
  
  float xp_old_3 = xp_old + dx;
  float yp_old_3 = yp_old - dy;
  cout << phi_heading << endl;
  cout << xp_old << "  " << yp_old << endl;
  cout << xp_old_1 << "  " << yp_old_1 << endl;
  cout << xp_old_2 << "  " << yp_old_2 << endl;
  cout << xp_old_3 << "  " << yp_old_3 << endl;
  
  // Query if xp_old_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_3 >= xsize + delta_x / 2.0){
      xp_old_3 -= xsize;
  }

  if (xp_old_3 < 0.0 + delta_x / 2.0){
      xp_old_3 += xsize;
  }
  
  // Query if yp_old_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_3 >= ysize + delta_y / 2.0){
      yp_old_3 -= ysize;
  }

  if (yp_old_3 < 0.0 + delta_y / 2.0){
      yp_old_3 += ysize;
  }

  // Determine relevant indices next to xp_old_3,yp_old_3, int()
  // reduced the input value to the next lower integer value
  // relevant indice  k stay equal from the previous calculation for w
  i = int(xp_old_3 / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_old_3 / delta_y - 0.5); // Relevant index for y-direction

  // Determie w at xp_old_3, y_old_3, z_old position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_old_3 / delta_x - i - 0.5); // w_value at x=xp_old_3 and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_old_3 / delta_x - i - 0.5); // w_value at xp_old_3 and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_old_3 / delta_y) - j - 0.5); // w_value at x= xp_old_3 and y= yp_old_3 and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_old_3 / delta_x - i - 0.5); // w_value at x=xp_old and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_old_3 / delta_x - i - 0.5); // w_value at x=xp_old_3 and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_old_3 / delta_y) - j- 0.5); // w_value at x= xp_old_3 and y= yp_old_3 and z=(k+1)*delta_z

  w_windspeed_3 = help3 + (help6 - help3) *
  ((zp_old / delta_z) - k); // w_Wert at x= xp_old_3 and y= yp_old_3 and z=zp_old


  
  
  // Sixth: wind at tail fin and wing tips for r-component
  // First v-component at wing tips
  // Query if xp_old_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_1 >= xsize + delta_x / 2.0){
      xp_old_1 -= xsize;
  }

  if (xp_old_1 < 0.0 + delta_x / 2.0){
      xp_old_1 += xsize;
  }
  
  // Query if yp_old_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_1 >= ysize){
      yp_old_1 -= ysize;
  }

  if (yp_old_1 < 0.0){
      yp_old_1 += ysize;
  }

  // Query if zp_old leaves the boundary
  if (zp_old >= zsize - delta_z/2.0){
      zp_old = zsize - (delta_z/2.0 + 0.000001);
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old_1 / delta_x - 0.5); //Relevant index for x-direction
  j = int(yp_old_1 / delta_y);     //Relevant index for y-direction
  k = int(zp_old   / delta_z + 0.5); //Relevant index for z-direction

  // Determie v at right wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_old_1 / delta_x - i - 0.5); //v_value at x=xp_old_1 and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_old_1 / delta_x - i - 0.5); //v_value at xp_old_1 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old_1 / delta_y) - j); // v_value at x= xp_old_1 and y= yp_old_1 and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_old_1 / delta_x - i - 0.5); //v_value at x=xp_old_1 and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_old_1 / delta_x - i - 0.5); //v_value at xp_old_1 and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old_1 / delta_y) - j); // v_value at x= xp_old_1 and y= yp_old_1 and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      v_windspeed_1 = help3 + (help6 - help3) *
     (zp_old / (0.5 * delta_z)); // v_value at x= xp_old_1 and y= yp_old_3 and z=zp_old
  }
  else {
      v_windspeed_1 = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // v_value at x= xp_old_1 and y= yp_old_3 and z=zp_old
  }  

  
  
  
  // Query if xp_old_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_2 >= xsize + delta_x / 2.0){
      xp_old_2 -= xsize;
  }
  
  if (xp_old_2 < 0.0 + delta_x / 2.0){
      xp_old_2 += xsize;
  }
  
  // Query if yp_old_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_2 >= ysize){
      yp_old_2 -= ysize;
  }

  if (yp_old_2 < 0.0){
      yp_old_2 += ysize;
  }
  
  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old_2 / delta_x - 0.5); //Relevant index for x-direction
  j = int(yp_old_2 / delta_y);     //Relevant index for y-direction
  
  // Determie v at left wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_old_2 / delta_x - i - 0.5); //v_value at x=xp_old_2 and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_old_2 / delta_x - i - 0.5); //v_value at xp_old_2 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old_2 / delta_y) - j); // v_value at x= xp_old_2 and y= yp_old_2 and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_old_2 / delta_x - i - 0.5); //v_value at x=xp_old_2 and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_old_2 / delta_x - i - 0.5); //v_value at xp_old_2 and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old_2 / delta_y) - j); // v_value at x= xp_old_2 and y= yp_old_2 and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      v_windspeed_2 = help3 + (help6 - help3) *
     (zp_old / (0.5 * delta_z)); // v_value at x= xp_old_2 and y= yp_old_3 and z=zp_old
  }
  else {
      v_windspeed_2 = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // v_value at x= xp_old_2 and y= yp_old_3 and z=zp_old
  }
  
  
  
  
  //Second u-component at wing tips
  // Query if xp_old_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_1 >= xsize){
      xp_old_1 -= xsize;
  }

  if (xp_old_1 < 0.0){
      xp_old_1 += xsize;
  }
  
  // Query if yp_old_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_1 >= ysize + delta_y / 2.0){
      yp_old_1 -= ysize;
  }

  if (yp_old_1 < 0.0 + delta_y / 2.0){
      yp_old_1 += ysize;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old_1 / delta_x);       //Relevant index for x-direction
  j = int(yp_old_1 / delta_y - 0.5); //Relevant index for y-direction

  // Determie u at right wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_old_1 / delta_x - i); //u_value at x=xp_old_1 and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_old_1 / delta_x - i); //u_value at xp_old_1 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old_1 / delta_y) - j - 0.5); // u_value at x= xp_old_1 and y= yp_old_1 and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_old_1 / delta_x - i); //u_value at x=xp_old_1 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_old_1 / delta_x - i); //u_value at xp_old_1 and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old_1 / delta_y) - j - 0.5); // u_value at x= xp_old_1 and y= yp_old_1 and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      u_windspeed_1 = help3 + (help6 - help3) *
      (zp_old / (0.5 * delta_z)); // u_value at x= xp_old_1 and y= yp_old_1 and z=zp_old
  }
  else
      u_windspeed_1 = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // u_value at x= xp_old_1 and y= yp_old_1 and z=zp_old
 
     
     
     
  // Query if xp_old_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_2 >= xsize){
      xp_old_2 -= xsize;
  }
  
  if (xp_old_2 < 0.0){
      xp_old_2 += xsize;
  }

  // Query if yp_old_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_2 >= ysize + delta_y / 2.0){
      yp_old_2 -= ysize;
  }

  if (yp_old_2 < 0.0 + delta_y / 2.0){
      yp_old_2 += ysize;
  }
  
  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old_2 / delta_x);       //Relevant index for x-direction
  j = int(yp_old_2 / delta_y - 0.5); //Relevant index for y-direction
  
  // Determie u at left wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_old_2 / delta_x - i); //u_value at x=xp_old_2 and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_old_2 / delta_x - i); //u_value at xp_old_2 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old_2 / delta_y) - j - 0.5); // u_value at x= xp_old_2 and y= yp_old_2 and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_old_2 / delta_x - i); //u_value at x=xp_old_2 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_old_2 / delta_x - i); //u_value at xp_old_2 and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old_2 / delta_y) - j - 0.5); // u_value at x= xp_old_2 and y= yp_old_2 and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      u_windspeed_2 = help3 + (help6 - help3) *
      (zp_old / (0.5 * delta_z)); // u_value at x= xp_old_2 and y= yp_old_2 and z=zp_old
  }
  else
      u_windspeed_2 = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // u_value at x= xp_old_2 and y= yp_old_2 and z=zp_old
     
  
     
     
  // Third u-component at fin position
  // Query if xp_old_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_3 >= xsize){
      xp_old_3 -= xsize;
  }

  if (xp_old_3 < 0.0){
      xp_old_3 += xsize;
  }

  // Query if yp_old_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_3 >= ysize + delta_y / 2.0){
      yp_old_3 -= ysize;
  }
  if (yp_old_3 < 0.0 + delta_y / 2.0){
      yp_old_3 += ysize;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old_3 / delta_x);       //Relevant index for x-direction
  j = int(yp_old_3 / delta_y - 0.5); //Relevant index for y-direction

  // Determie u at fin position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_old_3 / delta_x - i); //u_value at x=xp_old_3 and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_old_3 / delta_x - i); //u_value at xp_old_3 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old_3 / delta_y) - j - 0.5); // u_value at x= xp_old_3 and y= yp_old_3 and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_old_3 / delta_x - i); //u_value at x=xp_old_3 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_old_3 / delta_x - i); //u_value at xp_old_3 and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old_3 / delta_y) - j - 0.5); // u_value at x= xp_old_3 and y= yp_old_3 and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      u_windspeed_3 = help3 + (help6 - help3) *
      (zp_old / (0.5 * delta_z)); // u_value at x= xp_old_3 and y= yp_old_3 and z=zp_old
  }
  else
      u_windspeed_3 = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // u_value at x= xp_old_3 and y= yp_old_3 and z=zp_old

     
     
     
  // Fourth v-component at fin position
  // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_old_3 >= xsize + delta_x / 2.0){
      xp_old_3 -= xsize;
  }

  if (xp_old_3 < 0.0 + delta_x / 2.0){
      xp_old_3 += xsize;
  }

  // Query if yp_old_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_old_3 >= ysize){
      yp_old_3 -= ysize;
  }

  if (yp_old_3 < 0.0){
      yp_old_3 += ysize;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_old_3 / delta_x - 0.5); //Relevant index for x-direction
  j = int(yp_old_3 / delta_y);     //Relevant index for y-direction

  // Determie v at fin position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_old_3 / delta_x - i - 0.5); //v_value at x=xp_old_3 and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_old_3 / delta_x - i - 0.5); //v_value at xp_old_3 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_old_3 / delta_y) - j); // v_value at x= xp_old_3 and y= yp_old_3 and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_old_3 / delta_x - i - 0.5); //v_value at x=xp_old_3 and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_old_3 / delta_x - i - 0.5); //v_value at xp_old_3 and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_old_3 / delta_y) - j); // v_value at x= xp_old_3 and y= yp_old_3 and z=(k+1)*delta_z-delta_z/2

  if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      v_windspeed_3 = help3 + (help6 - help3) *
     (zp_old / (0.5 * delta_z)); // v_value at x= xp_old_3 and y= yp_old_3 and z=zp_old
  }
  else {
      v_windspeed_3 = help3 + (help6 - help3) *
     ((zp_old / delta_z) - k + 0.5); // v_value at x= xp_old_3 and y= yp_old_3 and z=zp_old
  }

  
  

  cout << "*** Calculating horinzontal mean wind values, like current, top and ground wind" << endl;


  if (windmodel == "PALM3D_homogeneous"){
    
    // Horizontal mean of the u- and v-component for each height. Only Neccessary for homogeneous ground
    // Horizontal meann of w-component is by definition zero
    float u_sum;
    float v_sum;

    for (k = 0; k < dimz; k = k+1){

        for (j = 0; j < dimy; j = j+1){

          for (i = 0; i < dimx; i = i+1){

              u_sum += u_palm[k][j][i];
              v_sum += v_palm[k][j][i];
          }
      }
      u_xyav_palm[k]= u_sum / (dimx * dimy);
      v_xyav_palm[k]= v_sum / (dimx * dimy);

      u_sum = 0;
      v_sum = 0;
    }

  
    // Determine ground wind for the homogeneous case (equal to the mean horizontal wind velocity in 20ft=6.096m heigth)
    k = int(6.096 / delta_z + 0.5);


    u_xyav_palm_20ft = u_xyav_palm[k] + (u_xyav_palm[k+1] - u_xyav_palm[k])
                           * (6.096 / delta_z - k + 0.5);

    v_xyav_palm_20ft = v_xyav_palm[k] + (v_xyav_palm[k+1] - v_xyav_palm[k])
                           * (6.096 / delta_z - k + 0.5);

    // Later on it is written to the property "/environment/config/boundary/entry[0]/wind-speed-kt"
    windspeed_ground = sqrt(pow(u_xyav_palm_20ft,2) + pow(v_xyav_palm_20ft,2)) * MPS2KT;
  
    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_ground = atan2(u_xyav_palm_20ft,v_xyav_palm_20ft);
    if (winddir_ground < 0){
        winddir_ground = (winddir_ground + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_ground =  winddir_ground * (180.0/M_PI) + 180.0;
    }
    
    
    // Determine upper wind (equal to the mean horizontal wind velocity at zu_max=zsize-delta_z/2)
    windspeed_top = sqrt(pow(u_xyav_palm[NZ-1],2) + pow(v_xyav_palm[NZ-1],2)) * MPS2KT;

    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_top = atan2(u_xyav_palm[NZ-1],v_xyav_palm[NZ-1]);
    if (winddir_top < 0){
        winddir_top = (winddir_top + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_top =  winddir_top * (180.0/M_PI) + 180.0;
    }
    
  
    // Linear interpolation of the horizontal mean of the u and v-component to the actual flight height
    // Query if zp_old leaves the boundary
    if (zp_old >= zsize - delta_z/2.0){
        zp_old = zsize - (delta_z/2.0 + 0.000001);
    }

    // Relevant index for z-direction
    k = int(zp_old / delta_z + 0.5);

    if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
        u_xyav_windspeed_current = u_xyav_palm[k] + (u_xyav_palm[k+1] -
        u_xyav_palm[k]) * (zp_old / (0.5 * delta_z));

        v_xyav_windspeed_current = v_xyav_palm[k] + (v_xyav_palm[k+1] -
        v_xyav_palm[k]) * (zp_old / (0.5 * delta_z));
    }
    else{

        u_xyav_windspeed_current = u_xyav_palm[k] + (u_xyav_palm[k+1] -
        u_xyav_palm[k]) * (zp_old / delta_z - k + 0.5);

        v_xyav_windspeed_current = v_xyav_palm[k] + (v_xyav_palm[k+1] -
        v_xyav_palm[k]) * (zp_old / delta_z - k + 0.5);
    }

    // Mean wind speed of the w-component is in the homogeneous case 0
    w_xyav_windspeed_current = 0.0;
    
    // Mean horizontal wind velocity for the current height
    // windspeed_current (/environment/wind-speed-kt) and winddir (/environment/wind-from-heading-deg) 
    // are used to calculate the properties /environment/wind-from-north-fps and /environment/wind-from-east-fps via
    // /environment/wind-from-east-fps = sin(winddir)*windspeed_current and 
    // /environment/wind-from-north-fps = cos(winddir)*windspeed_current
    // In JSBSim.cxx the vector vWindNED (averaged wind) is set with the help of the three einvironment wind properties
    windspeed_current_ms = sqrt(pow(u_xyav_windspeed_current,2) + pow(v_xyav_windspeed_current,2));

    // Calculate current horizontal averaged windspeed in knots
    windspeed_current_kt = windspeed_current_ms * MPS2KT; // Later on, windspeed_current is written to "/environment/wind-speed-kt"
    windspeed_current_fps = windspeed_current_ms * M2FT;   // For calculating the FG properties /environment/wind-from-...-fps to 
                                                           // set the vWindNED.
                                                           
    // Mean horizontal wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir = atan2(u_xyav_windspeed_current,v_xyav_windspeed_current);
    if (winddir < 0){
        winddir = (winddir + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir =  winddir * (180.0/M_PI) + 180.0;
    }
    
    // Calculate FG wind to set vWindNED
    wind_from_north = cos(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_east  = sin(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_down  = w_xyav_windspeed_current*M2FT; 
    
  }

  

  
  if (windmodel == "PALM3D_heterogeneous"){

    // Determine ground wind for the heterogeneous case (equal to the time averaged wind velocity in 20ft=6.096m heigth)    
    // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_old >= xsize){
        xp_old -= xsize;
    }

    if (xp_old < 0.0){
        xp_old += xsize;
    }

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize + delta_y / 2.0){
        yp_old -= ysize;
    }

    if (yp_old < 0.0 + delta_y / 2.0){
        yp_old += ysize;
    }
    
    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_old / delta_x);       // Relevant index for x-direction
    j = int(yp_old / delta_y - 0.5); // Relevant index for y-direction
    k = int(6.096 / delta_z + 0.5);  // Relevant index for z-direction
    
    // Determie u_av at the ground of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = u_av_palm[k][j][i] + (u_av_palm[k][j][i+1] - u_av_palm[k][j][i]) *
    (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

    help2 = u_av_palm[k][j+1][i] + (u_av_palm[k][j+1][i+1] - u_av_palm[k][j+1][i]) *
    (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=k*delta_z-delta_z/2

    help4 = u_av_palm[k+1][j][i] + (u_av_palm[k+1][j][i+1] - u_av_palm[k+1][j][i]) *
    (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help5 = u_av_palm[k+1][j+1][i] + (u_av_palm[k+1][j+1][i+1] - u_av_palm[k+1][j+1][i]) *
    (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=(k+1)*delta_z-delta_z/2

    u_xyav_palm_20ft = help3 + (help6 - help3) *
    ((6.096 / delta_z) - k + 0.5); // u_value at x= xp_old and y= yp_old and z=6.096m


    // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_old >= xsize + delta_x / 2.0){
        xp_old -= xsize;
    }

    if (xp_old < 0.0 + delta_x / 2.0){
        xp_old += xsize;
    }

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize){
        yp_old -= ysize;
    }

    if (yp_old < 0.0){
        yp_old += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_old / delta_x - 0.5); // Relevant index for x-direction
    j = int(yp_old / delta_y);       // Relevant index for y-direction 
    
    // Determie v_av at the ground of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = v_av_palm[k][j][i] + (v_av_palm[k][j][i+1] - v_av_palm[k][j][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=k*delta_z-delta_z/2

    help2 = v_av_palm[k][j+1][i] + (v_av_palm[k][j+1][i+1] - v_av_palm[k][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=k*delta_z-delta_z/2

    help4 = v_av_palm[k+1][j][i] + (v_av_palm[k+1][j][i+1] - v_av_palm[k+1][j][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

    help5 = v_av_palm[k+1][j+1][i] + (v_av_palm[k+1][j+1][i+1] - v_av_palm[k+1][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=(k+1)*delta_z-delta_z/2

    v_xyav_palm_20ft = help3 + (help6 - help3) *
    ((6.096 / delta_z) - k + 0.5); // v_value at x= xp_old and y= yp_old and z=6.096m

    
    // Later on it is written to the property "/environment/config/boundary/entry[0]/wind-speed-kt"
    windspeed_ground = sqrt(pow(u_xyav_palm_20ft,2) + pow(v_xyav_palm_20ft,2)) * MPS2KT;
  
    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_ground = atan2(u_xyav_palm_20ft,v_xyav_palm_20ft);
    if (winddir_ground < 0){
        winddir_ground = (winddir_ground + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_ground =  winddir_ground * (180.0/M_PI) + 180.0;
    }
    
    
    // Determine upper wind (equal to the mean horizontal wind velocity at zu_max=zsize-delta_z/2)
    // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_old >= xsize){
        xp_old -= xsize;
    }

    if (xp_old < 0.0){
        xp_old += xsize;
    }

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize + delta_y / 2.0){
        yp_old -= ysize;
    }

    if (yp_old < 0.0 + delta_y / 2.0){
        yp_old += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_old / delta_x);       // Relevant index for x-direction
    j = int(yp_old / delta_y - 0.5); // Relevant index for y-direction

    // Determie u_av at the top of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = u_av_palm[NZ-1][j][i] + (u_av_palm[NZ-1][j][i+1] - u_av_palm[NZ-1][j][i]) *
    (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=NZ-1*delta_z-delta_z/2

    help2 = u_av_palm[NZ-1][j+1][i] + (u_av_palm[NZ-1][j+1][i+1] - u_av_palm[NZ-1][j+1][i]) *
    (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y and z=NZ-1*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=NZ-1*delta_z-delta_z/2
   
    u_top = help3;
    
    
    // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_old >= xsize + delta_x / 2.0){
        xp_old -= xsize;
    }

    if (xp_old < 0.0 + delta_x / 2.0){
        xp_old += xsize;
    }

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize){
        yp_old -= ysize;
    }

    if (yp_old < 0.0){
        yp_old += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_old / delta_x - 0.5); // Relevant index for x-direction
    j = int(yp_old / delta_y);       // Relevant index for y-direction 
    
    // Determie v_av at the top of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = v_av_palm[NZ-1][j][i] + (v_av_palm[NZ-1][j][i+1] - v_av_palm[NZ-1][j][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=NZ-1*delta_z-delta_z/2

    help2 = v_av_palm[NZ-1][j+1][i] + (v_av_palm[NZ-1][j+1][i+1] - v_av_palm[NZ-1][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=NZ-1*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=NZ-1*delta_z-delta_z/2
    
    v_top = help3;
  
    
    windspeed_top = sqrt(pow(u_top,2) + pow(v_top,2)) * MPS2KT;

    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_top = atan2(u_top,v_top);
    if (winddir_top < 0){
        winddir_top = (winddir_top + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_top =  winddir_top * (180.0/M_PI) + 180.0;
    }

    
    // Linear interpolation of the local mean of the u and v-component to the actual flight position
    // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_old >= xsize){
        xp_old -= xsize;
    }

    if (xp_old < 0.0){
        xp_old += xsize;
    }

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize + delta_y / 2.0){
        yp_old -= ysize;
    }

    if (yp_old < 0.0 + delta_y / 2.0){
        yp_old += ysize;
    }
    
    // Query if zp_old leaves the boundary
    if (zp_old >= zsize - delta_z/2.0){
        zp_old = zsize - (delta_z/2.0 + 0.000001);
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_old / delta_x);       // Relevant index for x-direction
    j = int(yp_old / delta_y - 0.5); // Relevant index for y-direction
    k = int(zp_old / delta_z + 0.5); // Relevant index for z-direction
    
    // Determie u_av at aircraft position making a tri-linear interpolation (first x, then y, then z)
    help1 = u_av_palm[k][j][i] + (u_av_palm[k][j][i+1] - u_av_palm[k][j][i]) *
    (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

    help2 = u_av_palm[k][j+1][i] + (u_av_palm[k][j+1][i+1] - u_av_palm[k][j+1][i]) *
    (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=k*delta_z-delta_z/2

    help4 = u_av_palm[k+1][j][i] + (u_av_palm[k+1][j][i+1] - u_av_palm[k+1][j][i]) *
    (xp_old / delta_x - i); // u_value at x=xp_old and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help5 = u_av_palm[k+1][j+1][i] + (u_av_palm[k+1][j+1][i+1] - u_av_palm[k+1][j+1][i]) *
    (xp_old / delta_x - i); // u_value at xp_old and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_old / delta_y) - j - 0.5); // u_value at x= xp_old and y= yp_old and z=(k+1)*delta_z-delta_z/2

    if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                                 // at least for the u- and v-component
       u_xyav_windspeed_current = help3 + (help6 - help3) *
       (zp_old / (0.5 * delta_z)); // u_value at x= xp_old and y= yp_old and z=zp_old
    }
    else{
       u_xyav_windspeed_current= help3 + (help6 - help3) *
       ((zp_old / delta_z) - k + 0.5); // u_value at x= xp_old and y= yp_old and z=zp_old
    }

    
    // Query if xp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_old >= xsize + delta_x / 2.0){
        xp_old -= xsize;
    }

    if (xp_old < 0.0 + delta_x / 2.0){
        xp_old += xsize;
    }

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize){
        yp_old -= ysize;
    }

    if (yp_old < 0.0){
        yp_old += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_old / delta_x - 0.5); // Relevant index for x-direction
    j = int(yp_old / delta_y);       // Relevant index for y-direction
    
    // Determie v_av at aircraft position making a tri-linear interpolation (first x, then y, then z)
    help1 = v_av_palm[k][j][i] + (v_av_palm[k][j][i+1] - v_av_palm[k][j][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=k*delta_z-delta_z/2

    help2 = v_av_palm[k][j+1][i] + (v_av_palm[k][j+1][i+1] - v_av_palm[k][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=k*delta_z-delta_z/2

    help4 = v_av_palm[k+1][j][i] + (v_av_palm[k+1][j][i+1] - v_av_palm[k+1][j][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at x=xp_old and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

    help5 = v_av_palm[k+1][j+1][i] + (v_av_palm[k+1][j+1][i+1] - v_av_palm[k+1][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // v_value at xp_old and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_old / delta_y) - j); // v_value at x= xp_old and y= yp_old and z=(k+1)*delta_z-delta_z/2

    if (zp_old < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                                 // at least for the u- and v-component
       v_xyav_windspeed_current = help3 + (help6 - help3) *
       (zp_old / (0.5 * delta_z)); // v_value at x= xp_old and y= yp_old and z=zp_old
    }
    else {
       v_xyav_windspeed_current = help3 + (help6 - help3) *
       ((zp_old / delta_z) - k + 0.5); // v_value at x= xp_old and y= yp_old and z=zp_old
    }
 

    // Query if yp_old leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_old >= ysize + delta_y / 2.0){
        yp_old -= ysize;
    }

    if (yp_old < 0.0 + delta_y / 2.0){
        yp_old += ysize;
    }

    // Query if zp_old leaves the boundary
    if (zp_old > zsize){
        zp_old = zsize - 0.000001;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    j = int(yp_old / delta_y - 0.5); // Relevant index for y-direction
    k = int(zp_old / delta_z);       // Relevant index for z-direction

    // Determie w_av at aircraft position making a tri-linear interpolation (first x, then y, then z)
    help1 = w_av_palm[k][j][i] + (w_av_palm[k][j][i+1] - w_av_palm[k][j][i]) *
    (xp_old / delta_x - i - 0.5); // w_value at x=xp_old and y=j*delta_y+delta_y/2 and z=k*delta_z

    help2 = w_av_palm[k][j+1][i] + (w_av_palm[k][j+1][i+1] - w_av_palm[k][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // w_value at xp_old and (j+1)*delta_y+delta_y/2 and z=k*delta_z

    help3 = help1 + (help2 - help1) *
    ((yp_old / delta_y) - j - 0.5); // w_value at x= xp_old and y= yp_old and z=k*delta_z

    help4 = w_av_palm[k+1][j][i] + (w_av_palm[k+1][j][i+1] - w_av_palm[k+1][j][i]) *
    (xp_old / delta_x - i - 0.5); //w_value at x=xp_old and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

    help5 = w_av_palm[k+1][j+1][i] + (w_av_palm[k+1][j+1][i+1] - w_av_palm[k+1][j+1][i]) *
    (xp_old / delta_x - i - 0.5); // w_value at x=xp_old and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

    help6 = help4 + (help5 - help4) *
    ((yp_old / delta_y) - j- 0.5); // w_value at x= xp_old and y= yp_old and z=(k+1)*delta_z

    w_xyav_windspeed_current = help3 + (help6 - help3) *
    ((zp_old / delta_z) - k); // w_Wert at x= xp_old and y= yp_old and z=zp_old  
    
    
    // Mean horizontal wind velocity for the current height
    // windspeed_current (/environment/wind-speed-kt) and winddir (/environment/wind-from-heading-deg) 
    // are used to calculate the properties /environment/wind-from-north-fps and /environment/wind-from-east-fps via
    // /environment/wind-from-east-fps = sin(winddir)*windspeed_current and 
    // /environment/wind-from-north-fps = cos(winddir)*windspeed_current  
    windspeed_current_ms = sqrt(pow(u_xyav_windspeed_current,2) + pow(v_xyav_windspeed_current,2));

    // Calculate current horizontal averaged windspeed in knots
    windspeed_current_kt = windspeed_current_ms * MPS2KT; // Later on, windspeed_current is written to "/environment/wind-speed-kt"
    windspeed_current_fps = windspeed_current_ms * M2FT;   // For calculating the FG properties /environment/wind-from-...-fps to 
                                                           // set the vWindNED.
                                                           
    // Mean horizontal wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir = atan2(u_xyav_windspeed_current,v_xyav_windspeed_current);
    if (winddir < 0){
        winddir = (winddir + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir =  winddir * (180.0/M_PI) + 180.0;
    }
    
    // Calculate FG wind to set vWindNED
    wind_from_north = cos(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_east  = sin(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_down  = w_xyav_windspeed_current*M2FT; 
  
  } 
  

  Setnode = fgGetNode("/environment/config/boundary/entry/wind-speed-kt",true);
  Setnode->setFloatValue(windspeed_ground);

  Setnode = fgGetNode("fdm/jsbsim/atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps",true);
  Setnode->setFloatValue(windspeed_ground);

  Setnode = fgGetNode("/environment/config/boundary/entry/wind-from-heading-deg",true);
  Setnode->setFloatValue(winddir_ground);

  Setnode = fgGetNode("/environment/config/boundary/entry/wind-speed-kt-top",true);
  Setnode->setFloatValue(windspeed_top);

  Setnode = fgGetNode("/environment/config/boundary/entry/wind-from-heading-deg-top",true);
  Setnode->setFloatValue(winddir_top);

  Setnode = fgGetNode("/environment/wind-speed-kt",false);
  Setnode->setFloatValue(windspeed_current_kt);

  Setnode = fgGetNode("/environment/wind-from-heading-deg",false);
  Setnode->setFloatValue(winddir);
  
  Setnode = fgGetNode("/environment/wind-from-down-fps",false);
  Setnode->setFloatValue(w_xyav_windspeed_current);

  
  cout << "*** Calculating horizontal mean wind values successfully" << endl;

  
  // A step which is done in JSBSim.cxx for the default version: Set variable vWindNED  
  SetWindNED(-wind_from_north, -wind_from_east, -wind_from_down);

  
  // Determine turbulence signal in NED frame
    
  // linear
  vTurbulenceNED(eNorth) = (v_windspeed - v_xyav_windspeed_current)*M2FT; // In PALM is a positiv v-component in north direction (south wind)
  vTurbulenceNED(eEast)  = (u_windspeed - u_xyav_windspeed_current)*M2FT; // In PALM is a positiv u-comonent in east direction (west wind)
  vTurbulenceNED(eDown)  = -(w_windspeed - w_xyav_windspeed_current)*M2FT;                     // Horizontal average of w-component in PALM is 0; positiv w in PALM
                                                                        //means upwards therefore change the sign.

  /*
  // First Method: Determine gradients in wind frame where the x-axis is parallel to the mean wind direction
  // and transform them into the NED frame. positive w_windspeed means upward in JSBSim downward 
  vTurbPQR(eP) = - ((-1*w_windspeed_1 - (-1)*w_windspeed_2) / in.wingspan); // roll rate in NED frame,
                                                                    // w-component is equal in WF and NED frame,
                                                                    // > 0 means a right/clockwise turn (right wing goes down)
                                                                    
  // rotational (Calculate Gradients first in wind frame and transform turbulent components afterwards into NED frame)
  vTurbPQR(eQ) = (-1*w_windspeed - (-1)*w_windspeed_3) / vtailarm;  // pitch rate in NED frame,
                                                            // w-component is equal in WF and NED frame,
                                                            // > 0 means nose upward

  // u_windspeed_2_WF/u_windspeed_1_WF is equal to xi_u in the JSBSim calculations
  u_windspeed_2_WF = v_windspeed_2 * cos(psiw) -
                     u_windspeed_2 * sin(psiw);
                     
  u_windspeed_1_WF = v_windspeed_1 * cos(psiw) -
                     u_windspeed_1 * sin(psiw);
  
  r1g  = ((u_windspeed_2_WF - u_windspeed_1_WF) / in.wingspan);              
                                                       
  // v_windspeed_3_WF/v_windspeed_WF is equal to xi_v in the JSBSim calculations
  v_windspeed_3_WF = v_windspeed_3 * sin(psiw) +
                     u_windspeed_3 * cos(psiw);

  v_windspeed_WF   = v_windspeed * sin(psiw) +
                     u_windspeed * cos(psiw);

  r2g  = ((v_windspeed_WF - v_windspeed_3_WF) / vtailarm);
  
  vTurbPQR(eR)  = r1g + r2g;                                        // yaw rate in NED frame,
                                                                    // r-component is equal in WF and NED frame,
                                                                    // > 0 means yaw to the right side (nose goes right)*/

  //Second method: Determine Gradients directely in the body frame
  // 1 means north-, 2 east- and 3 down-direction
  
  // wind vector at CG, already rotated for heterogeneous case(see above)
  windspeed_NED(1) = v_windspeed; 
  windspeed_NED(2) = u_windspeed;
  windspeed_NED(3) = -w_windspeed;
  
  // wind vector at left wing
  windspeed_1_NED(1) = v_windspeed_1;  
  windspeed_1_NED(2) = u_windspeed_1; 
  windspeed_1_NED(3) = -w_windspeed_1;
  
  // wind vector at right wing 
  windspeed_2_NED(1) = v_windspeed_2;  
  windspeed_2_NED(2) = u_windspeed_2; 
  windspeed_2_NED(3) = -w_windspeed_3;

  // wind vector at fin 
  windspeed_3_NED(1) = v_windspeed_3;  
  windspeed_3_NED(2) = u_windspeed_3; 
  windspeed_3_NED(3) = -w_windspeed_3;
  
  // Transform wind vector to body axis
  windspeed_body   = in.Tl2b*windspeed_NED;
  windspeed_1_body = in.Tl2b*windspeed_1_NED; 
  windspeed_2_body = in.Tl2b*windspeed_2_NED; 
  windspeed_3_body = in.Tl2b*windspeed_3_NED;

  vTurbPQR(eP) = - ((windspeed_1_body(3) - windspeed_2_body(3)) / in.wingspan);
  vTurbPQR(eQ) = (windspeed_body(3) - windspeed_3_body(3)) / vtailarm; 
  r1g          = ((windspeed_2_body(1) - windspeed_1_body(1)) / in.wingspan); 
  r2g          = ((windspeed_body(2) - windspeed_3_body(2)) / vtailarm);

  vTurbPQR(eR)  = r1g + r2g;
  
  vTurbPQR = in.Tb2l*vTurbPQR;
   

  cout << "*** Determine wind data for the initial position successfully" << endl;


  elapsed_time = 0.0;


  return 0;

  
  

  } catch(NcException& e)
    {
      e.what();
      cout<<"FAILURE*************************************"<<endl;
      return NC_ERR;
    }
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void FGWinds::PALMLoop()
{
  // Timestep of JSBSim model
  Getnode = fgGetNode("sim/model-hz",false);
  delta_t = 1.0 / Getnode->getFloatValue();

  // Get the ground speed from the property tree
  Getnode = fgGetNode("fdm/jsbsim/velocities/vg-fps",false);
  V_g =  Getnode->getFloatValue() * FT2M;
  
  // Get the velocities in the local coordinate frame
  Getnode = fgGetNode("/fdm/jsbsim/velocities/v-east-fps",false);
  U = Getnode->getDoubleValue() * FT2M;
  Getnode = fgGetNode("/fdm/jsbsim/velocities/v-north-fps",false);
  V = Getnode->getDoubleValue() * FT2M;
 

  // Determination of the new position after one time step in the PALM coordinate system
  if (windmodel == "PALM3D_heterogeneous"){
     U_rotated = cos(rotationangle * M_PI/180.0) * U + sin(rotationangle * M_PI/180.0) * V; 
     V_rotated = -sin(rotationangle * M_PI/180.0) * U + cos(rotationangle * M_PI/180.0) * V; 
     xp_new = xp_old + U_rotated * delta_t; // x is defined in west-east direction
     yp_new = yp_old + V_rotated * delta_t; // y is defined in south-north direction
  }
  else if (windmodel == "PALM3D_homogeneous"){
     xp_new = xp_old + U * delta_t;  
     yp_new = yp_old + V * delta_t;
  }
  zp_new = in.DistanceAGL * FT2M; // position/altitude-ft = Höhe über NN
  

  // Determination of the covered distance (only horizontal)
  x_distance = xp_new - xp_old;
  y_distance = yp_new  - yp_old;
  distance_h = distance_h + sqrt(pow(x_distance,2)+pow(y_distance,2));

  Setnode = fgGetNode("/local-weather/PALM/covered-distance",true);
  Setnode->setFloatValue(distance_h);  
  
  
  // First: u-component at C.G.
  // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new >= xsize){
      xp_new -= xsize;
  }

  if (xp_new < 0.0){
      xp_new += xsize;
  }

  Setnode = fgGetNode("/local-weather/PALM/x-position",true);
  Setnode->setFloatValue(xp_new);
  
  // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new >= ysize + delta_y / 2.0){
      yp_new -= ysize;
  }

  if (yp_new < 0.0 + delta_y / 2.0){
      yp_new += ysize;
  }

  // Query if zp_new leaves the boundary
  if (zp_new >= zsize - delta_z/2.0){
      zp_new = zsize - (delta_z/2.0 + 0.000001);
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new / delta_x);       // Relevant index for x-direction
  j = int(yp_new / delta_y - 0.5); // Relevant index for y-direction
  k = int(zp_new / delta_z + 0.5); // Relevant index for z-direction

  // Determie u at aircraft position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
     u_windspeed = help3 + (help6 - help3) *
     (zp_new / (0.5 * delta_z)); // u_value at x= xp_new and y= yp_new and z=zp_new
  }
  else
     u_windspeed = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // u_value at x= xp_new and y= yp_new and z=zp_new

     
     

  // Second: v-component at C.G.
  // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new >= xsize + delta_x / 2.0){
      xp_new -= xsize;
  }

  if (xp_new < 0.0 + delta_x / 2.0){
      xp_new += xsize;
  }

  // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new >= ysize){
      yp_new -= ysize;
  }

  if (yp_new < 0.0){
      yp_new += ysize;
  }
  
  Setnode = fgGetNode("/local-weather/PALM/y-position",true);
  Setnode->setFloatValue(yp_new);
  
  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_new / delta_y);       // Relevant index for y-direction

  // Determie v at aircraft position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
     v_windspeed = help3 + (help6 - help3) *
     (zp_new / (0.5 * delta_z)); // v_value at x= xp_new and y= yp_new and z=zp_new
  }
  else {
     v_windspeed = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // v_value at x= xp_new and y= yp_new and z=zp_new
  }

  
  

  // Third: w-component at C.G.
  // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new >= xsize + delta_x / 2.0){
      xp_new -= xsize;
  }

  if (xp_new < 0.0 + delta_x / 2.0){
      xp_new += xsize;
  }

  // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new >= ysize + delta_y / 2.0){
      yp_new -= ysize;
  }

  if (yp_new < 0.0 + delta_y / 2.0){
      yp_new += ysize;
  }

  // Query if zp_new leaves the boundary
  if (zp_new > zsize){
      zp_new = zsize - 0.000001;
  }

  Setnode = fgGetNode("/local-weather/PALM/z-position",true);
  Setnode->setFloatValue(zp_new);
  
  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_new / delta_y - 0.5); // Relevant index for y-direction
  k = int(zp_new / delta_z);       // Relevant index for z-direction

  // Determie w at aircraft position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_new / delta_x - i - 0.5); // w_value at x=xp_new and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_new / delta_x - i - 0.5); // w_value at xp_new and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_new / delta_y) - j - 0.5); // w_value at x= xp_new and y= yp_new and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_new / delta_x - i - 0.5); //w_value at x=xp_new and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_new / delta_x - i - 0.5); // w_value at x=xp_new and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_new / delta_y) - j- 0.5); // w_value at x= xp_new and y= yp_new and z=(k+1)*delta_z

  w_windspeed = help3 + (help6 - help3) *
  ((zp_new / delta_z) - k); // w_Wert at x= xp_new and y= yp_new and z=zp_new

  
  

  // Fourth: wind at wing tips for p
  // indices _1,_2,_3 stands for the positions described in the 4 point model of
  // B. Etkin: "Turbulent wind and its effect on flight" (1981).

  // Get the heading of the aircraft
  Getnode = fgGetNode("orientation/heading-deg",false);
  phi_heading = Getnode->getFloatValue(); // North direction = 0, East direction = 90 ....
  
  // Determine position of the right and left wing tip ((Assumption:
  // Aircraft roll angle is 0. The wings are aligned along the horinzontal plane and have
  // no vertical displacement.)
  alpha = -(phi_heading - 360) * (M_PI/180.0);
  dx = cos(alpha) * in.wingspan/2.0 * FT2M;  
  dy = sin(alpha) * in.wingspan/2.0 * FT2M;

  // w at right wing
  xp_new_1 = xp_new + dx;
  yp_new_1 = yp_new + dy;  

  // Query if xp_new_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_1 >= xsize + delta_x / 2.0){
      xp_new_1 -= xsize;
  }

  if (xp_new_1 < 0.0 + delta_x / 2.0){
      xp_new_1 += xsize;
  }
  
  // Query if yp_new_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_1 >= ysize + delta_y / 2.0){
      yp_new_1 -= ysize;
  }

  if (yp_new_1 < 0.0 + delta_y / 2.0){
      yp_new_1 += ysize;
  }
  
  // Determine relevant indices next to xp_new_1,yp_new_1, int()
  // reduced the input value to the next lower integer value
  // relevant indices j and k stay equal from the previous calculation for w
  i = int(xp_new_1 / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_new_1 / delta_y - 0.5); // Relevant index for y-direction

  // Determie w at xp_new_1, y_new, z_new position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_new_1 / delta_x - i - 0.5); // w_value at x=xp_new_1 and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_new_1 / delta_x - i - 0.5); // w_value at xp_new_1 and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_new_1 / delta_y) - j - 0.5); // w_value at x= xp_new_1 and y= yp_new_1 and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_new_1 / delta_x - i - 0.5); // w_value at x=xp_new_1 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_new_1 / delta_x - i - 0.5); // w_value at x=xp_new_1 and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_new_1 / delta_y) - j- 0.5); // w_value at x= xp_new_1 and y= yp_new_1 and z=(k+1)*delta_z

  w_windspeed_1 = help3 + (help6 - help3) *
  ((zp_new / delta_z) - k); // w_Wert at x= xp_new_1 and y= yp_new_1 and z=zp_new

  
  
  
  // w at left wing
  xp_new_2 = xp_new - dx;
  yp_new_2 = yp_new - dy;
  
  // Query if xp_new_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_2 >= xsize + delta_x / 2.0){
      xp_new_2 -= xsize;
  }

  if (xp_new_2 < 0.0 + delta_x / 2.0){
      xp_new_2 += xsize;
  }
  
  // Query if yp_new_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_2 >= ysize + delta_y / 2.0){
      yp_new_2 -= ysize;
  }

  if (yp_new_2 < 0.0 + delta_y / 2.0){
      yp_new_2 += ysize;
  }
  
  // Determine relevant indices next to xp_new_2,yp_new_2, int()
  // reduced the input value to the next lower integer value
  // relevant indices j and k stay equal from the previous calculation for w
  i = int(xp_new_2 / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_new_2 / delta_y - 0.5); // Relevant index for y-direction

  // Determie w at xp_new_2, y_new_2, z_new position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_new_2 / delta_x - i - 0.5); // w_value at x=xp_new_2 and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_new_2 / delta_x - i - 0.5); //w_value at xp_new_2 and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_new_2 / delta_y) - j - 0.5); // w_value at x= xp_new_2 and y= yp_new_2 and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_new_2 / delta_x - i - 0.5); // w_value at x=xp_new_2 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_new_2 / delta_x - i - 0.5); // w_value at x=xp_new_2 and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_new_2 / delta_y) - j- 0.5); // w_value at x= xp_new_2 and y= yp_new_2 and z=(k+1)*delta_z

  w_windspeed_2 = help3 + (help6 - help3) *
  ((zp_new / delta_z) - k); // w_Wert at x= xp_new_2 and y= yp_new_2 and z=zp_new

  
  

  // Fifth: wind at tail fin for q-component
  // Determine position of the fin ((Assumption: Aircraft pitch angle is 0. 
  // The longitudinal axis is aligned along the horinzontal plane and have
  // no vertical displacement.)
  dx = sin(alpha) * vtailarm;  
  dy = cos(alpha) * vtailarm;
  
  xp_new_3 = xp_new + dx;
  yp_new_3 = yp_new - dy;

  // Query if xp_new_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_3 >= xsize + delta_x / 2.0){
      xp_new_3 -= xsize;
  }

  if (xp_new_3 < 0.0 + delta_x / 2.0){
      xp_new_3 += xsize;
  }
  
  // Query if yp_new_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_3 >= ysize + delta_y / 2.0){
      yp_new_3 -= ysize;
  }

  if (yp_new_3 < 0.0 + delta_y / 2.0){
      yp_new_3 += ysize;
  }

  // Determine relevant indices next to xp_new_3,yp_new_3, int()
  // reduced the input value to the next lower integer value
  // relevant indice  k stay equal from the previous calculation for w
  i = int(xp_new_3 / delta_x - 0.5); // Relevant index for x-direction
  j = int(yp_new_3 / delta_y - 0.5); // Relevant index for y-direction

  // Determie w at xp_new_3, y_new_3, z_new position making a tri-linear interpolation (first x, then y, then z)
  help1 = w_palm[k][j][i] + (w_palm[k][j][i+1] - w_palm[k][j][i]) *
  (xp_new_3 / delta_x - i - 0.5); // w_value at x=xp_new_3 and y=j*delta_y+delta_y/2 and z=k*delta_z

  help2 = w_palm[k][j+1][i] + (w_palm[k][j+1][i+1] - w_palm[k][j+1][i]) *
  (xp_new_3 / delta_x - i - 0.5); // w_value at xp_new_3 and (j+1)*delta_y+delta_y/2 and z=k*delta_z

  help3 = help1 + (help2 - help1) *
  ((yp_new_3 / delta_y) - j - 0.5); // w_value at x= xp_new and y= yp_new_3 and z=k*delta_z

  help4 = w_palm[k+1][j][i] + (w_palm[k+1][j][i+1] - w_palm[k+1][j][i]) *
  (xp_new_3 / delta_x - i - 0.5); // w_value at x=xp_new_3 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

  help5 = w_palm[k+1][j+1][i] + (w_palm[k+1][j+1][i+1] - w_palm[k+1][j+1][i]) *
  (xp_new_3 / delta_x - i - 0.5); // w_value at x=xp_new_3 and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

  help6 = help4 + (help5 - help4) *
  ((yp_new_3 / delta_y) - j- 0.5); // w_value at x= xp_new and y= yp_new_3 and z=(k+1)*delta_z

  w_windspeed_3 = help3 + (help6 - help3) *
  ((zp_new / delta_z) - k); // w_Wert at x= xp_new_3 and y= yp_new_3 and z=zp_new


  
  
  // Sixth: wind at tail fin and wing tips for r-component
  // First v-component at wing tips
  // Query if xp_new_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_1 >= xsize + delta_x / 2.0){
      xp_new_1 -= xsize;
  }

  if (xp_new_1 < 0.0 + delta_x / 2.0){
      xp_new_1 += xsize;
  }
  
  // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_1 >= ysize){
      yp_new_1 -= ysize;
  }

  if (yp_new_1 < 0.0){
      yp_new_1 += ysize;
  }

  // Query if zp_new leaves the boundary
  if (zp_new >= zsize - delta_z/2.0){
      zp_new = zsize - (delta_z/2.0 + 0.000001);
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new_1 / delta_x - 0.5); //Relevant index for x-direction
  j = int(yp_new_1   / delta_y);     //Relevant index for y-direction
  k = int(zp_new   / delta_z + 0.5); //Relevant index for z-direction

  // Determie v at right wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_new_1 / delta_x - i - 0.5); //v_value at x=xp_new_1 and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_new_1 / delta_x - i - 0.5); //v_value at xp_new_1 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new_1 / delta_y) - j); // v_value at x= xp_new_1 and y= yp_new_1 and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_new_1 / delta_x - i - 0.5); //v_value at x=xp_new_1 and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_new_1 / delta_x - i - 0.5); //v_value at xp_new_1 and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new_1 / delta_y) - j); // v_value at x= xp_new_1 and y= yp_new_1 and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      v_windspeed_1 = help3 + (help6 - help3) *
     (zp_new / (0.5 * delta_z)); // v_value at x= xp_new_1 and y= yp_new_1 and z=zp_new
  }
  else {
      v_windspeed_1 = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // v_value at x= xp_new_1 and y= yp_new_1 and z=zp_new
  }

  
  
  
  // Query if xp_new_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_2 >= xsize + delta_x / 2.0){
      xp_new_2 -= xsize;
  }
  
  if (xp_new_2 < 0.0 + delta_x / 2.0){
      xp_new_2 += xsize;
  }
  
  // Query if yp_new_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_2 >= ysize){
      yp_new_2 -= ysize;
  }

  if (yp_new_2 < 0.0){
      yp_new_2 += ysize;
  }
  
  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new_2 / delta_x - 0.5); //Relevant index for x-direction
  j = int(yp_new_2 / delta_y);     //Relevant index for y-direction

  // Determie v at left wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_new_2 / delta_x - i - 0.5); //v_value at x=xp_new_2 and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_new_2 / delta_x - i - 0.5); //v_value at xp_new_2 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new_2 / delta_y) - j); // v_value at x= xp_new_2 and y= yp_new_2 and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_new_2 / delta_x - i - 0.5); //v_value at x=xp_new_2 and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_new_2 / delta_x - i - 0.5); //v_value at xp_new_2 and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new_2 / delta_y) - j); // v_value at x= xp_new_2 and y= yp_new_2 and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      v_windspeed_2 = help3 + (help6 - help3) *
     (zp_new / (0.5 * delta_z)); // v_value at x= xp_new_2 and y= yp_new_2 and z=zp_new
  }
  else {
      v_windspeed_2 = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // v_value at x= xp_new_2 and y= yp_new_2 and z=zp_new
  }
  
  
  
  
  //Second u-component at wing tips
  // Query if xp_new_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_1 >= xsize){
      xp_new_1 -= xsize;
  }

  if (xp_new_1 < 0.0){
      xp_new_1 += xsize;
  }
  
  // Query if yp_new_1 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_1 >= ysize + delta_y / 2.0){
      yp_new_1 -= ysize;
  }

  if (yp_new_1 < 0.0 + delta_y / 2.0){
      yp_new_1 += ysize;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new_1 / delta_x);       //Relevant index for x-direction
  j = int(yp_new_1 / delta_y - 0.5); //Relevant index for y-direction

  // Determie u at right wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_new_1 / delta_x - i); //u_value at x=xp_new_1 and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_new_1 / delta_x - i); //u_value at xp_new_1 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new_1 / delta_y) - j - 0.5); // u_value at x= xp_new_1 and y= yp_new_1 and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_new_1 / delta_x - i); //u_value at x=xp_new_1 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_new_1 / delta_x - i); //u_value at xp_new_1 and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new_1 / delta_y) - j - 0.5); // u_value at x= xp_new_1 and y= yp_new_1 and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      u_windspeed_1 = help3 + (help6 - help3) *
      (zp_new / (0.5 * delta_z)); // u_value at x= xp_new_1 and y= yp_new_1 and z=zp_new
  }
  else
      u_windspeed_1 = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // u_value at x= xp_new_1 and y= yp_new_1 and z=zp_new
  
     
     
     
  // Query if xp_new_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_2 >= xsize){
      xp_new_2 -= xsize;
  }
  
  if (xp_new_2 < 0.0){
      xp_new_2 += xsize;
  }

  // Query if yp_new_2 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_2 >= ysize + delta_y / 2.0){
      yp_new_2 -= ysize;
  }

  if (yp_new_2 < 0.0 + delta_y / 2.0){
      yp_new_2 += ysize;
  }
  
  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new_2 / delta_x);       //Relevant index for x-direction
  j = int(yp_new_2 / delta_y - 0.5); //Relevant index for y-direction

  // Determie u at left wing position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_new_2 / delta_x - i); //u_value at x=xp_new_2 and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_new_2 / delta_x - i); //u_value at xp_new_2 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new_2 / delta_y) - j - 0.5); // u_value at x= xp_new_2 and y= yp_new_2 and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_new_2 / delta_x - i); //u_value at x=xp_new_2 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_new_2 / delta_x - i); //u_value at xp_new_2 and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new_2 / delta_y) - j - 0.5); // u_value at x= xp_new_2 and y= yp_new_2 and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      u_windspeed_2 = help3 + (help6 - help3) *
      (zp_new / (0.5 * delta_z)); // u_value at x= xp_new_2 and y= yp_new_2 and z=zp_new
  }
  else
      u_windspeed_2 = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // u_value at x= xp_new_2 and y= yp_new_2 and z=zp_new
     
     

     
  //Third u-component at fin position
  // Query if xp_new_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_3 >= xsize){
      xp_new_3 -= xsize;
  }

  if (xp_new_3 < 0.0){
      xp_new_3 += xsize;
  }

  // Query if yp_new_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_3 >= ysize + delta_y / 2.0){
      yp_new_3 -= ysize;
  }

  if (yp_new_3 < 0.0 + delta_y / 2.0){
      yp_new_3 += ysize;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new_3 / delta_x);       //Relevant index for x-direction
  j = int(yp_new_3 / delta_y - 0.5); //Relevant index for y-direction

  // Determie u at aircraft fin position making a tri-linear interpolation (first x, then y, then z)
  help1 = u_palm[k][j][i] + (u_palm[k][j][i+1] - u_palm[k][j][i]) *
  (xp_new_3 / delta_x - i); //u_value at x=xp_new_3 and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

  help2 = u_palm[k][j+1][i] + (u_palm[k][j+1][i+1] - u_palm[k][j+1][i]) *
  (xp_new_3 / delta_x - i); //u_value at xp_new_3 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new_3 / delta_y) - j - 0.5); // u_value at x= xp_new_3 and y= yp_new_3 and z=k*delta_z-delta_z/2

  help4 = u_palm[k+1][j][i] + (u_palm[k+1][j][i+1] - u_palm[k+1][j][i]) *
  (xp_new_3 / delta_x - i); //u_value at x=xp_new_3 and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help5 = u_palm[k+1][j+1][i] + (u_palm[k+1][j+1][i+1] - u_palm[k+1][j+1][i]) *
  (xp_new_3 / delta_x - i); //u_value at xp_new_3 and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new_3 / delta_y) - j - 0.5); // u_value at x= xp_new_3 and y= yp_new_3 and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      u_windspeed_3 = help3 + (help6 - help3) *
      (zp_new / (0.5 * delta_z)); // u_value at x= xp_new_3 and y= yp_new_3 and z=zp_new
  }
  else
      u_windspeed_3 = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // u_value at x= xp_new_3 and y= yp_new_3 and z=zp_new

     
     
     
  // Fourth v-component at fin position
  // Query if xp_new_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (xp_new_3 >= xsize + delta_x / 2.0){
      xp_new_3 -= xsize;
  }

  if (xp_new_3 < 0.0 + delta_x / 2.0){
      xp_new_3 += xsize;
  }

  // Query if yp_new_3 leaves the boundary (cyclic horizontal boundary conditions)
  if (yp_new_3 >= ysize){
      yp_new_3 -= ysize;
  }

  if (yp_new_3 < 0.0){
      yp_new_3 += ysize;
  }

  // Determine relevant indeces next to the aircraft position, int()
  // reduced the input value to the next lower integer value
  i = int(xp_new_3 / delta_x - 0.5); //Relevant index for x-direction
  j = int(yp_new_3 / delta_y);     //Relevant index for y-direction

  // Determie v at aircraft fin position making a tri-linear interpolation (first x, then y, then z)
  help1 = v_palm[k][j][i] + (v_palm[k][j][i+1] - v_palm[k][j][i]) *
  (xp_new_3 / delta_x - i - 0.5); //v_value at x=xp_new_3 and y=j*delta_y and z=k*delta_z-delta_z/2

  help2 = v_palm[k][j+1][i] + (v_palm[k][j+1][i+1] - v_palm[k][j+1][i]) *
  (xp_new_3 / delta_x - i - 0.5); //v_value at xp_new_3 and (j+1)*delta_y and z=k*delta_z-delta_z/2

  help3 = help1 + (help2 - help1) *
  ((yp_new_3 / delta_y) - j); // v_value at x= xp_new_3 and y= yp_new_3 and z=k*delta_z-delta_z/2

  help4 = v_palm[k+1][j][i] + (v_palm[k+1][j][i+1] - v_palm[k+1][j][i]) *
  (xp_new_3 / delta_x - i - 0.5); //v_value at x=xp_new_3 and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

  help5 = v_palm[k+1][j+1][i] + (v_palm[k+1][j+1][i+1] - v_palm[k+1][j+1][i]) *
  (xp_new_3 / delta_x - i - 0.5); //v_value at xp_new_3 and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

  help6 = help4 + (help5 - help4) *
  ((yp_new_3 / delta_y) - j); // v_value at x= xp_new_3 and y= yp_new_3 and z=(k+1)*delta_z-delta_z/2

  if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
      v_windspeed_3 = help3 + (help6 - help3) *
     (zp_new / (0.5 * delta_z)); // v_value at x= xp_new_3 and y= yp_new_3 and z=zp_new
  }
  else {
      v_windspeed_3 = help3 + (help6 - help3) *
     ((zp_new / delta_z) - k + 0.5); // v_value at x= xp_new_3 and y= yp_new_3 and z=zp_new
  }

  
  
  
  // Calculation is necessary, although windspeed_ground and winddir_ground are constant since the 
  // initialization. It's because of the fact that the Nasal scripts will overwrite the values and  
  // properties, respectively, after clicking/press the button okay in the weather condition menu
  if (windmodel == "PALM3D_homogeneous"){
  
    // Determine ground wind for the homogeneous case (equal to the mean horizontal wind velocity in 20ft=6.096m heigth)
    k = int(6.096 / delta_z + 0.5);


    u_xyav_palm_20ft = u_xyav_palm[k] + (u_xyav_palm[k+1] - u_xyav_palm[k])
                           * (6.096 / delta_z - k + 0.5);

    v_xyav_palm_20ft = v_xyav_palm[k] + (v_xyav_palm[k+1] - v_xyav_palm[k])
                           * (6.096 / delta_z - k + 0.5);

    // Later on it is written to the property "/environment/config/boundary/entry[0]/wind-speed-kt"
    windspeed_ground = sqrt(pow(u_xyav_palm_20ft,2) + pow(v_xyav_palm_20ft,2)) * MPS2KT;
  
    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_ground = atan2(u_xyav_palm_20ft,v_xyav_palm_20ft);
    if (winddir_ground < 0){
        winddir_ground = (winddir_ground + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_ground =  winddir_ground * (180.0/M_PI) + 180.0;
    }
    
    
    // Linear interpolation of the horizontal mean of the u and v-component to the actual flight height
    // Query if zp_new leaves the boundary
    if (zp_new >= zsize - delta_z/2.0){
        zp_new = zsize - (delta_z/2.0 + 0.000001);
    }

    // Relevant index for z-direction
    k = int(zp_new / delta_z + 0.5);

    if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                               // at least for the u- and v-component
        u_xyav_windspeed_current = u_xyav_palm[k] + (u_xyav_palm[k+1] -
        u_xyav_palm[k]) * (zp_new / (0.5 * delta_z));

        v_xyav_windspeed_current = v_xyav_palm[k] + (v_xyav_palm[k+1] -
        v_xyav_palm[k]) * (zp_new / (0.5 * delta_z));
    }
    else{

        u_xyav_windspeed_current = u_xyav_palm[k] + (u_xyav_palm[k+1] -
        u_xyav_palm[k]) * (zp_new / delta_z - k + 0.5);

        v_xyav_windspeed_current = v_xyav_palm[k] + (v_xyav_palm[k+1] -
        v_xyav_palm[k]) * (zp_new / delta_z - k + 0.5);
    }

    // Mean wind speed of the w-component is in the homogeneous case 0
    w_xyav_windspeed_current = 0.0;
    
    // Mean horizontal wind velocity for the current height
    // windspeed_current (/environment/wind-speed-kt) and winddir (/environment/wind-from-heading-deg) 
    // are used to calculate the properties /environment/wind-from-north-fps and /environment/wind-from-east-fps via
    // /environment/wind-from-east-fps = sin(winddir)*windspeed_current and 
    // /environment/wind-from-north-fps = cos(winddir)*windspeed_current
    // In JSBSim.cxx the vector vWindNED (averaged wind) is set with the help of the three einvironment wind properties
    windspeed_current_ms = sqrt(pow(u_xyav_windspeed_current,2) + pow(v_xyav_windspeed_current,2));

    // Calculate current horizontal averaged windspeed in knots
    windspeed_current_kt = windspeed_current_ms * MPS2KT; // Later on, windspeed_current is written to "/environment/wind-speed-kt"
    windspeed_current_fps = windspeed_current_ms * M2FT;   // For calculating the FG properties /environment/wind-from-...-fps to 
                                                           // set the vWindNED.
                                                           
    // Mean horizontal wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir = atan2(u_xyav_windspeed_current,v_xyav_windspeed_current);
    if (winddir < 0){
        winddir = (winddir + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir =  winddir * (180.0/M_PI) + 180.0;
    }
          
    // Calculate FG wind to set vWindNED
    wind_from_north = cos(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_east  = sin(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_down  = w_xyav_windspeed_current*M2FT; 
    
  }
  
  
  
  
  if (windmodel == "PALM3D_heterogeneous"){

    // Determine ground wind for the heterogeneous case (equal to the time averaged wind velocity in 20ft=6.096m heigth)    
    // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_new >= xsize){
        xp_new -= xsize;
    }

    if (xp_new < 0.0){
        xp_new += xsize;
    }

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize + delta_y / 2.0){
        yp_new -= ysize;
    }

    if (yp_new < 0.0 + delta_y / 2.0){
        yp_new += ysize;
    }
    
    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_new / delta_x);       // Relevant index for x-direction
    j = int(yp_new / delta_y - 0.5); // Relevant index for y-direction
    k = int(6.096 / delta_z + 0.5);  // Relevant index for z-direction
    

    // Determie u_av at the ground of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = u_av_palm[k][j][i] + (u_av_palm[k][j][i+1] - u_av_palm[k][j][i]) *
    (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

    help2 = u_av_palm[k][j+1][i] + (u_av_palm[k][j+1][i+1] - u_av_palm[k][j+1][i]) *
    (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=k*delta_z-delta_z/2

    help4 = u_av_palm[k+1][j][i] + (u_av_palm[k+1][j][i+1] - u_av_palm[k+1][j][i]) *
    (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help5 = u_av_palm[k+1][j+1][i] + (u_av_palm[k+1][j+1][i+1] - u_av_palm[k+1][j+1][i]) *
    (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=(k+1)*delta_z-delta_z/2

    u_xyav_palm_20ft = help3 + (help6 - help3) *
    ((6.096 / delta_z) - k + 0.5); // u_value at x= xp_new and y= yp_new and z=6.096m


    // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_new >= xsize + delta_x / 2.0){
        xp_new -= xsize;
    }

    if (xp_new < 0.0 + delta_x / 2.0){
        xp_new += xsize;
    }

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize){
        yp_new -= ysize;
    }

    if (yp_new < 0.0){
        yp_new += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_new / delta_x - 0.5); // Relevant index for x-direction
    j = int(yp_new / delta_y);       // Relevant index for y-direction 
    
    // Determie v_av at the ground of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = v_av_palm[k][j][i] + (v_av_palm[k][j][i+1] - v_av_palm[k][j][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=k*delta_z-delta_z/2

    help2 = v_av_palm[k][j+1][i] + (v_av_palm[k][j+1][i+1] - v_av_palm[k][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=k*delta_z-delta_z/2

    help4 = v_av_palm[k+1][j][i] + (v_av_palm[k+1][j][i+1] - v_av_palm[k+1][j][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

    help5 = v_av_palm[k+1][j+1][i] + (v_av_palm[k+1][j+1][i+1] - v_av_palm[k+1][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=(k+1)*delta_z-delta_z/2

    v_xyav_palm_20ft = help3 + (help6 - help3) *
    ((6.096 / delta_z) - k + 0.5); // v_value at x= xp_new and y= yp_new and z=6.096m
     
    
    // Later on it is written to the property "/environment/config/boundary/entry[0]/wind-speed-kt"
    windspeed_ground = sqrt(pow(u_xyav_palm_20ft,2) + pow(v_xyav_palm_20ft,2)) * MPS2KT;

    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_ground = atan2(u_xyav_palm_20ft,v_xyav_palm_20ft);
    if (winddir_ground < 0){
        winddir_ground = (winddir_ground + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_ground =  winddir_ground * (180.0/M_PI) + 180.0;
    }
    
    
    // Determine upper wind (equal to the mean horizontal wind velocity at zu_max=zsize-delta_z/2)
    // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_new >= xsize){
        xp_new -= xsize;
    }

    if (xp_new < 0.0){
        xp_new += xsize;
    }

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize + delta_y / 2.0){
        yp_new -= ysize;
    }

    if (yp_new < 0.0 + delta_y / 2.0){
        yp_new += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_new / delta_x);       // Relevant index for x-direction
    j = int(yp_new / delta_y - 0.5); // Relevant index for y-direction

    // Determie u_av at the top of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = u_av_palm[NZ-1][j][i] + (u_av_palm[NZ-1][j][i+1] - u_av_palm[NZ-1][j][i]) *
    (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=NZ-1*delta_z-delta_z/2

    help2 = u_av_palm[NZ-1][j+1][i] + (u_av_palm[NZ-1][j+1][i+1] - u_av_palm[NZ-1][j+1][i]) *
    (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y and z=NZ-1*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=NZ-1*delta_z-delta_z/2
    
    u_top = help3;
    
    
    // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_new >= xsize + delta_x / 2.0){
        xp_new -= xsize;
    }

    if (xp_new < 0.0 + delta_x / 2.0){
        xp_new += xsize;
    }

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize){
        yp_new -= ysize;
    }

    if (yp_new < 0.0){
        yp_new += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_new / delta_x - 0.5); // Relevant index for x-direction
    j = int(yp_new / delta_y);       // Relevant index for y-direction 
    
    // Determie v_av at the top of the model domain from the aircraft position
    // making a tri-linear interpolation (first x, then y, then z)
    help1 = v_av_palm[NZ-1][j][i] + (v_av_palm[NZ-1][j][i+1] - v_av_palm[NZ-1][j][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=NZ-1*delta_z-delta_z/2

    help2 = v_av_palm[NZ-1][j+1][i] + (v_av_palm[NZ-1][j+1][i+1] - v_av_palm[NZ-1][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=NZ-1*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=NZ-1*delta_z-delta_z/2

    
    v_top = help3;

    
    windspeed_top = sqrt(pow(u_top,2) + pow(v_top,2)) * MPS2KT;

    // Mean horizontal ground wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir_top = atan2(u_top,v_top);
    if (winddir_top < 0){
        winddir_top = (winddir_top + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir_top =  winddir_top * (180.0/M_PI) + 180.0;
    }

    
    // Linear interpolation of the local mean of the u and v-component to the actual flight position
    // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_new >= xsize){
        xp_new -= xsize;
    }

    if (xp_new < 0.0){
        xp_new += xsize;
    }

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize + delta_y / 2.0){
        yp_new -= ysize;
    }

    if (yp_new < 0.0 + delta_y / 2.0){
        yp_new += ysize;
    }
    
    // Query if zp_new leaves the boundary
    if (zp_new >= zsize - delta_z/2.0){
        zp_new = zsize - (delta_z/2.0 + 0.000001);
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_new / delta_x);       // Relevant index for x-direction
    j = int(yp_new / delta_y - 0.5); // Relevant index for y-direction
    k = int(zp_new / delta_z + 0.5); // Relevant index for z-direction
    
    // Determie u_av at aircraft position making a tri-linear interpolation (first x, then y, then z)
    help1 = u_av_palm[k][j][i] + (u_av_palm[k][j][i+1] - u_av_palm[k][j][i]) *
    (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=k*delta_z-delta_z/2

    help2 = u_av_palm[k][j+1][i] + (u_av_palm[k][j+1][i+1] - u_av_palm[k][j+1][i]) *
    (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=k*delta_z-delta_z/2

    help4 = u_av_palm[k+1][j][i] + (u_av_palm[k+1][j][i+1] - u_av_palm[k+1][j][i]) *
    (xp_new / delta_x - i); // u_value at x=xp_new and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help5 = u_av_palm[k+1][j+1][i] + (u_av_palm[k+1][j+1][i+1] - u_av_palm[k+1][j+1][i]) *
    (xp_new / delta_x - i); // u_value at xp_new and (j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_new / delta_y) - j - 0.5); // u_value at x= xp_new and y= yp_new and z=(k+1)*delta_z-delta_z/2

    if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                                 // at least for the u- and v-component
       u_xyav_windspeed_current = help3 + (help6 - help3) *
       (zp_new / (0.5 * delta_z)); // u_value at x= xp_new and y= yp_new and z=zp_new
    }
    else{
       u_xyav_windspeed_current= help3 + (help6 - help3) *
       ((zp_new / delta_z) - k + 0.5); // u_value at x= xp_new and y= yp_new and z=zp_new
    }

    
    // Query if xp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (xp_new >= xsize + delta_x / 2.0){
        xp_new -= xsize;
    }

    if (xp_new < 0.0 + delta_x / 2.0){
        xp_new += xsize;
    }

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize){
        yp_new -= ysize;
    }

    if (yp_new < 0.0){
        yp_new += ysize;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    i = int(xp_new / delta_x - 0.5); // Relevant index for x-direction
    j = int(yp_new / delta_y);       // Relevant index for y-direction
    
    // Determie v_av at aircraft position making a tri-linear interpolation (first x, then y, then z)
    help1 = v_av_palm[k][j][i] + (v_av_palm[k][j][i+1] - v_av_palm[k][j][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=k*delta_z-delta_z/2

    help2 = v_av_palm[k][j+1][i] + (v_av_palm[k][j+1][i+1] - v_av_palm[k][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=k*delta_z-delta_z/2

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=k*delta_z-delta_z/2

    help4 = v_av_palm[k+1][j][i] + (v_av_palm[k+1][j][i+1] - v_av_palm[k+1][j][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at x=xp_new and y=j*delta_y and z=(k+1)*delta_z-delta_z/2

    help5 = v_av_palm[k+1][j+1][i] + (v_av_palm[k+1][j+1][i+1] - v_av_palm[k+1][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // v_value at xp_new and (j+1)*delta_y and z=(k+1)*delta_z-delta_z/2

    help6 = help4 + (help5 - help4) *
    ((yp_new / delta_y) - j); // v_value at x= xp_new and y= yp_new and z=(k+1)*delta_z-delta_z/2

    if (zp_new < delta_z / 2.0){ // Because the first and second vertical gridpoints are seperated from each other by delta_z/2,
                                 // at least for the u- and v-component
       v_xyav_windspeed_current = help3 + (help6 - help3) *
       (zp_new / (0.5 * delta_z)); // v_value at x= xp_new and y= yp_new and z=zp_new
    }
    else {
       v_xyav_windspeed_current = help3 + (help6 - help3) *
       ((zp_new / delta_z) - k + 0.5); // v_value at x= xp_new and y= yp_new and z=zp_new
    }
 

    // Query if yp_new leaves the boundary (cyclic horizontal boundary conditions)
    if (yp_new >= ysize + delta_y / 2.0){
        yp_new -= ysize;
    }

    if (yp_new < 0.0 + delta_y / 2.0){
        yp_new += ysize;
    }

    // Query if zp_new leaves the boundary
    if (zp_new > zsize){
        zp_new = zsize - 0.000001;
    }

    // Determine relevant indeces next to the aircraft position, int()
    // reduced the input value to the next lower integer value
    j = int(yp_new / delta_y - 0.5); // Relevant index for y-direction
    k = int(zp_new / delta_z);       // Relevant index for z-direction

    // Determie w_av at aircraft position making a tri-linear interpolation (first x, then y, then z)
    help1 = w_av_palm[k][j][i] + (w_av_palm[k][j][i+1] - w_av_palm[k][j][i]) *
    (xp_new / delta_x - i - 0.5); // w_value at x=xp_new and y=j*delta_y+delta_y/2 and z=k*delta_z

    help2 = w_av_palm[k][j+1][i] + (w_av_palm[k][j+1][i+1] - w_av_palm[k][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // w_value at xp_new and (j+1)*delta_y+delta_y/2 and z=k*delta_z

    help3 = help1 + (help2 - help1) *
    ((yp_new / delta_y) - j - 0.5); // w_value at x= xp_new and y= yp_new and z=k*delta_z

    help4 = w_av_palm[k+1][j][i] + (w_av_palm[k+1][j][i+1] - w_av_palm[k+1][j][i]) *
    (xp_new / delta_x - i - 0.5); //w_value at x=xp_new and y=j*delta_y+delta_y/2 and z=(k+1)*delta_z

    help5 = w_av_palm[k+1][j+1][i] + (w_av_palm[k+1][j+1][i+1] - w_av_palm[k+1][j+1][i]) *
    (xp_new / delta_x - i - 0.5); // w_value at x=xp_new and y=(j+1)*delta_y+delta_y/2 and z=(k+1)*delta_z

    help6 = help4 + (help5 - help4) *
    ((yp_new / delta_y) - j- 0.5); // w_value at x= xp_new and y= yp_new and z=(k+1)*delta_z

    w_xyav_windspeed_current = help3 + (help6 - help3) *
    ((zp_new / delta_z) - k); // w_Wert at x= xp_new and y= yp_new and z=zp_new  
    
    
    // Mean horizontal wind velocity for the current height
    // windspeed_current (/environment/wind-speed-kt) and winddir (/environment/wind-from-heading-deg) 
    // are used to calculate the properties /environment/wind-from-north-fps and /environment/wind-from-east-fps via
    // /environment/wind-from-east-fps = sin(winddir)*windspeed_current and 
    // /environment/wind-from-north-fps = cos(winddir)*windspeed_current  
    windspeed_current_ms = sqrt(pow(u_xyav_windspeed_current,2) + pow(v_xyav_windspeed_current,2));

    // Calculate current horizontal averaged windspeed in knots and fps
    windspeed_current_kt  = windspeed_current_ms * MPS2KT; // Later on, windspeed_current is written to "/environment/wind-speed-kt"
    windspeed_current_fps = windspeed_current_ms * M2FT;   // For calculating the FG properties /environment/wind-from-...-fps to 
                                                           // set the vWindNED. 
                                                      
    // Mean horizontal wind direction for FlightGear (north wind,in the south direction = 0°,
    // west wind, in the east direction = 270°)
    winddir = atan2(u_xyav_windspeed_current,v_xyav_windspeed_current);
    if (winddir < 0){
        winddir = (winddir + 2.0 * M_PI) * (180.0/M_PI) + 180.0;
    }
    else {
        winddir =  winddir * (180.0/M_PI) + 180.0;
    }

    // Calculate FG wind to set vWindNED
    wind_from_north = cos(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_east  = sin(winddir * M_PI/180.0) * windspeed_current_fps;
    wind_from_down  = w_xyav_windspeed_current*M2FT; 
    
    Setnode = fgGetNode("/environment/config/boundary/entry/wind-speed-kt-top",true);
    Setnode->setFloatValue(windspeed_top);

    Setnode = fgGetNode("/environment/config/boundary/entry/wind-from-heading-deg-top",true);
    Setnode->setFloatValue(winddir_top);
  
  }

  Setnode = fgGetNode("/environment/config/boundary/entry/wind-speed-kt",true);
  Setnode->setFloatValue(windspeed_ground);

  Setnode = fgGetNode("fdm/jsbsim/atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps",true);
  Setnode->setFloatValue(windspeed_ground);

  Setnode = fgGetNode("/environment/config/boundary/entry/wind-from-heading-deg",true);
  Setnode->setFloatValue(winddir_ground);

  Setnode = fgGetNode("/environment/wind-speed-kt",false);
  Setnode->setFloatValue(windspeed_current_kt);

  Setnode = fgGetNode("/environment/wind-from-heading-deg",false);
  Setnode->setFloatValue(winddir);
  
  Setnode = fgGetNode("/environment/wind-from-down-fps",false);
  Setnode->setFloatValue(w_xyav_windspeed_current);


  // A step which is done in JSBSim.cxx for the default version: Set variable vWindNED  
  SetWindNED(-wind_from_north, -wind_from_east, -wind_from_down);

  
  // Determine turbulence signal in NED frame
  
  // linear
  vTurbulenceNED(eNorth) = (v_windspeed - v_xyav_windspeed_current)*M2FT; // In PALM is a positiv v-component in north direction (south wind)
  vTurbulenceNED(eEast)  = (u_windspeed - u_xyav_windspeed_current)*M2FT; // In PALM is a positiv u-comonent in east direction (west wind)
  vTurbulenceNED(eDown)  = -(w_windspeed - w_xyav_windspeed_current)*M2FT; // Horizontal average of w-component in PALM is 0; positiv w in PALM
                                                                          //means upwards therefore change the sign.
                                                                          
  
  /* First Method: Determine gradients in wind frame where the x-axis is parallel to the mean wind direction
  // and transform them into the NED frame. positive w_windspeed means upward in JSBSim downward 
  vTurbPQR(eP) = - ((-1*w_windspeed_1 - (-1)*w_windspeed_2) / in.wingspan); // roll rate in NED frame,
                                                                    // w-component is equal in WF and NED frame,
                                                                    // > 0 means a right/clockwise turn (right wing goes down)
                                                                    
  // rotational (Calculate Gradients first in wind frame and transform turbulent components afterwards into NED frame)
  vTurbPQR(eQ) = (-1*w_windspeed - (-1)*w_windspeed_3) / vtailarm;  // pitch rate in NED frame,
                                                            // w-component is equal in WF and NED frame,
                                                            // > 0 means nose upward

  // u_windspeed_2_WF/u_windspeed_1_WF is equal to xi_u in the JSBSim calculations
  u_windspeed_2_WF = v_windspeed_2 * cos(psiw) -
                     u_windspeed_2 * sin(psiw);
                     
  u_windspeed_1_WF = v_windspeed_1 * cos(psiw) -
                     u_windspeed_1 * sin(psiw);
  
  r1g  = ((u_windspeed_2_WF - u_windspeed_1_WF) / in.wingspan);              
                                                       
  // v_windspeed_3_WF/v_windspeed_WF is equal to xi_v in the JSBSim calculations
  v_windspeed_3_WF = v_windspeed_3 * sin(psiw) +
                     u_windspeed_3 * cos(psiw);

  v_windspeed_WF   = v_windspeed * sin(psiw) +
                     u_windspeed * cos(psiw);

  r2g  = ((v_windspeed_WF - v_windspeed_3_WF) / vtailarm);
  
  vTurbPQR(eR)  = r1g + r2g;                                        // yaw rate in NED frame,
                                                                    // r-component is equal in WF and NED frame,
                                                                    // > 0 means yaw to the right side (nose goes right)*/
                                                                    
  //Second method: Determine Gradients directely in the body frame
  // Index 1 means north-, 2 east- and 3 down-direction
  
  // wind vector at CG, already rotated for heterogeneous case(see above)
  windspeed_NED(1) = v_windspeed; 
  windspeed_NED(2) = u_windspeed;
  windspeed_NED(3) = -w_windspeed;
  
  // wind vector at left wing
  windspeed_1_NED(1) = v_windspeed_1;  
  windspeed_1_NED(2) = u_windspeed_1; 
  windspeed_1_NED(3) = -w_windspeed_1;
  
  // wind vector at right wing
  windspeed_2_NED(1) = v_windspeed_2;  
  windspeed_2_NED(2) = u_windspeed_2; 
  windspeed_2_NED(3) = -w_windspeed_3;

  // wind vector at fin 
  windspeed_3_NED(1) = v_windspeed_3;  
  windspeed_3_NED(2) = u_windspeed_3; 
  windspeed_3_NED(3) = -w_windspeed_3;
  
  // Transform wind vector to body axis
  windspeed_body   = in.Tl2b*windspeed_NED;
  windspeed_1_body = in.Tl2b*windspeed_1_NED; 
  windspeed_2_body = in.Tl2b*windspeed_2_NED; 
  windspeed_3_body = in.Tl2b*windspeed_3_NED;

  vTurbPQR(eP) = - ((windspeed_1_body(3) - windspeed_2_body(3)) / in.wingspan);
  vTurbPQR(eQ) = (windspeed_body(3) - windspeed_3_body(3)) / vtailarm; 
  r1g          = ((windspeed_2_body(1) - windspeed_1_body(1)) / in.wingspan); 
  r2g          = ((windspeed_body(2) - windspeed_3_body(2)) / vtailarm);

  vTurbPQR(eR)  = r1g + r2g;
  
  vTurbPQR = in.Tb2l*vTurbPQR;

  
  xp_old = xp_new;
  yp_old = yp_new;
  
  
  elapsed_time += delta_t;
  Setnode = fgGetNode("local-weather/PALM/elapsed-time",false);
  Setnode->setFloatValue(elapsed_time);


}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void FGWinds::Turbulence(double h)
{        

  if (windmodel == "PALM3D_homogeneous" && local_weather_running == 1 && tile_type == "PALM" ||
      windmodel == "PALM3D_heterogeneous" && local_weather_running == 1 && tile_type == "PALM"){

     // Function to read initial PALM wind data
     if (input_PALM_data_flag == 0) {
         FT2M = 0.3048;
         M2FT = 1.0 / 0.3048;
         KT2MPS = 0.5144444444;
         MPS2KT = 1 / KT2MPS;
         vtailarm = 57.33 * FT2M;
         if (windmodel == "PALM3D_heterogeneous") { rotationangle = 62.0;}
         else {rotationangle = 0.0;}
         cout << "rotationangle: " << rotationangle << endl;
         InputPALM();
        }
     // # Begin the loop for determine the windfields each time the function
     // Turbulence is called (with JSBSim rate; default 120Hz)
     PALMLoop();
  }

  else {
      switch (turbType) {




      case ttCulp: {

        vTurbPQR(eP) = wind_from_clockwise;
        if (TurbGain == 0.0) return;

        // keep the inputs within allowable limts for this model
        if (TurbGain < 0.0) TurbGain = 0.0;
        if (TurbGain > 1.0) TurbGain = 1.0;
        if (TurbRate < 0.0) TurbRate = 0.0;
        if (TurbRate > 30.0) TurbRate = 30.0;
        if (Rhythmicity < 0.0) Rhythmicity = 0.0;
        if (Rhythmicity > 1.0) Rhythmicity = 1.0;

        // generate a sine wave corresponding to turbulence rate in hertz
        double time = FDMExec->GetSimTime();
        double sinewave = sin( time * TurbRate * 6.283185307 );

        double random = 0.0;
        if (target_time == 0.0) {
          strength = random = 1 - 2.0*(double(rand())/double(RAND_MAX));
          target_time = time + 0.71 + (random * 0.5);
        }
        if (time > target_time) {
          spike = 1.0;
          target_time = 0.0;
        }

        // max vertical wind speed in fps, corresponds to TurbGain = 1.0
        double max_vs = 40;

        vTurbulenceNED.InitMatrix();
        double delta = strength * max_vs * TurbGain * (1-Rhythmicity) * spike;

        // Vertical component of turbulence.
        vTurbulenceNED(eDown) = sinewave * max_vs * TurbGain * Rhythmicity;
        vTurbulenceNED(eDown)+= delta;
        if (in.DistanceAGL/in.wingspan < 3.0)
            vTurbulenceNED(eDown) *= in.DistanceAGL/in.wingspan * 0.3333;

        // Yaw component of turbulence.
        vTurbulenceNED(eNorth) = sin( delta * 3.0 );
        vTurbulenceNED(eEast) = cos( delta * 3.0 );

        // Roll component of turbulence. Clockwise vortex causes left roll.
        vTurbPQR(eP) += delta * 0.04;

        spike = spike * 0.9;
        break;
      }
      case ttMilspec:
      case ttTustin: {
 
        // an index of zero means turbulence is disabled
        // airspeed occurs as divisor in the code below
        if (probability_of_exceedence_index == 0 || in.V == 0) {
          vTurbulenceNED(eNorth) = vTurbulenceNED(eEast) = vTurbulenceNED(eDown) = 0.0;
          vTurbPQR(eP) = vTurbPQR(eQ) = vTurbPQR(eR) = 0.0;
          return;
        }

        // Turbulence model according to MIL-F-8785C (Flying Qualities of Piloted Aircraft)
        double b_w = in.wingspan, L_u, L_w, sig_u, sig_w;

          if (b_w == 0.) b_w = 30.;

        // clip height functions at 10 ft
        if (h <= 10.) h = 10;

        // Scale lengths L and amplitudes sigma as function of height
        if (h <= 1000) {
          L_u = h/pow(0.177 + 0.000823*h, 1.2); // MIL-F-8785c, Fig. 10, p. 55
          L_w = h;
          sig_w = 0.1*windspeed_at_20ft;
          sig_u = sig_w/pow(0.177 + 0.000823*h, 0.4); // MIL-F-8785c, Fig. 11, p. 56
        } else if (h <= 2000) {
          // linear interpolation between low altitude and high altitude models
          L_u = L_w = 1000 + (h-1000.)/1000.*750.;
          sig_u = sig_w = 0.1*windspeed_at_20ft
                        + (h-1000.)/1000.*(POE_Table->GetValue(probability_of_exceedence_index, h) - 0.1*windspeed_at_20ft);
        } else {
          L_u = L_w = 1750.; //  MIL-F-8785c, Sec. 3.7.2.1, p. 48
          sig_u = sig_w = POE_Table->GetValue(probability_of_exceedence_index, h);
        }

        // keep values from last timesteps
        // TODO maybe use deque?
        static double
          xi_u_km1 = 0, nu_u_km1 = 0,
          xi_v_km1 = 0, xi_v_km2 = 0, nu_v_km1 = 0, nu_v_km2 = 0,
          xi_w_km1 = 0, xi_w_km2 = 0, nu_w_km1 = 0, nu_w_km2 = 0,
          xi_p_km1 = 0, nu_p_km1 = 0,
          xi_q_km1 = 0, xi_r_km1 = 0;


        double
          T_V = in.totalDeltaT, // for compatibility of nomenclature
          sig_p = 1.9/sqrt(L_w*b_w)*sig_w, // Yeager1998, eq. (8)
          //sig_q = sqrt(M_PI/2/L_w/b_w), // eq. (14)
          //sig_r = sqrt(2*M_PI/3/L_w/b_w), // eq. (17)
          L_p = sqrt(L_w*b_w)/2.6, // eq. (10)
          tau_u = L_u/in.V, // eq. (6)
          tau_w = L_w/in.V, // eq. (3)
          tau_p = L_p/in.V, // eq. (9)
          tau_q = 4*b_w/M_PI/in.V, // eq. (13)
          tau_r =3*b_w/M_PI/in.V, // eq. (17)
          nu_u = GaussianRandomNumber(),
          nu_v = GaussianRandomNumber(),
          nu_w = GaussianRandomNumber(),
          nu_p = GaussianRandomNumber(),
          xi_u=0, xi_v=0, xi_w=0, xi_p=0, xi_q=0, xi_r=0;

        // values of turbulence NED velocities

        if (turbType == ttTustin) {
          // the following is the Tustin formulation of Yeager's report
          double
            omega_w = in.V/L_w, // hidden in nomenclature p. 3
            omega_v = in.V/L_u, // this is defined nowhere
            C_BL  = 1/tau_u/tan(T_V/2/tau_u), // eq. (19)
            C_BLp = 1/tau_p/tan(T_V/2/tau_p), // eq. (22)
            C_BLq = 1/tau_q/tan(T_V/2/tau_q), // eq. (24)
            C_BLr = 1/tau_r/tan(T_V/2/tau_r); // eq. (26)

          // all values calculated so far are strictly positive, except for
          // the random numbers nu_*. This means that in the code below, all
          // divisors are strictly positive, too, and no floating point
          // exception should occur.
          xi_u = -(1 - C_BL*tau_u)/(1 + C_BL*tau_u)*xi_u_km1
               + sig_u*sqrt(2*tau_u/T_V)/(1 + C_BL*tau_u)*(nu_u + nu_u_km1); // eq. (18)
          xi_v = -2*(sqr(omega_v) - sqr(C_BL))/sqr(omega_v + C_BL)*xi_v_km1
               - sqr(omega_v - C_BL)/sqr(omega_v + C_BL) * xi_v_km2
               + sig_u*sqrt(3*omega_v/T_V)/sqr(omega_v + C_BL)*(
                     (C_BL + omega_v/sqrt(3.))*nu_v
                   + 2/sqrt(3.)*omega_v*nu_v_km1
                   + (omega_v/sqrt(3.) - C_BL)*nu_v_km2); // eq. (20) for v
          xi_w = -2*(sqr(omega_w) - sqr(C_BL))/sqr(omega_w + C_BL)*xi_w_km1
               - sqr(omega_w - C_BL)/sqr(omega_w + C_BL) * xi_w_km2
               + sig_w*sqrt(3*omega_w/T_V)/sqr(omega_w + C_BL)*(
                     (C_BL + omega_w/sqrt(3.))*nu_w
                   + 2/sqrt(3.)*omega_w*nu_w_km1
                   + (omega_w/sqrt(3.) - C_BL)*nu_w_km2); // eq. (20) for w
          xi_p = -(1 - C_BLp*tau_p)/(1 + C_BLp*tau_p)*xi_p_km1
               + sig_p*sqrt(2*tau_p/T_V)/(1 + C_BLp*tau_p) * (nu_p + nu_p_km1); // eq. (21)
          xi_q = -(1 - 4*b_w*C_BLq/M_PI/in.V)/(1 + 4*b_w*C_BLq/M_PI/in.V) * xi_q_km1
               + C_BLq/in.V/(1 + 4*b_w*C_BLq/M_PI/in.V) * (xi_w - xi_w_km1); // eq. (23)
          xi_r = - (1 - 3*b_w*C_BLr/M_PI/in.V)/(1 + 3*b_w*C_BLr/M_PI/in.V) * xi_r_km1
               + C_BLr/in.V/(1 + 3*b_w*C_BLr/M_PI/in.V) * (xi_v - xi_v_km1); // eq. (25)

        } else if (turbType == ttMilspec) {
          // the following is the MIL-STD-1797A formulation
          // as cited in Yeager's report
          xi_u = (1 - T_V/tau_u)  *xi_u_km1 + sig_u*sqrt(2*T_V/tau_u)*nu_u;  // eq. (30)
          xi_v = (1 - 2*T_V/tau_u)*xi_v_km1 + sig_u*sqrt(4*T_V/tau_u)*nu_v;  // eq. (31)
          xi_w = (1 - 2*T_V/tau_w)*xi_w_km1 + sig_w*sqrt(4*T_V/tau_w)*nu_w;  // eq. (32)
          xi_p = (1 - T_V/tau_p)  *xi_p_km1 + sig_p*sqrt(2*T_V/tau_p)*nu_p;  // eq. (33)
          xi_q = (1 - T_V/tau_q)  *xi_q_km1 + M_PI/4/b_w*(xi_w - xi_w_km1);  // eq. (34)
          xi_r = (1 - T_V/tau_r)  *xi_r_km1 + M_PI/3/b_w*(xi_v - xi_v_km1);  // eq. (35)
        }

        // rotate by wind azimuth and assign the velocities
        double cospsi = cos(psiw), sinpsi = sin(psiw);
        vTurbulenceNED(eNorth) =  cospsi*xi_u + sinpsi*xi_v;
        vTurbulenceNED(eEast) = -sinpsi*xi_u + cospsi*xi_v;
        vTurbulenceNED(eDown) = xi_w;

        vTurbPQR(eP) =  cospsi*xi_p + sinpsi*xi_q;
        vTurbPQR(eQ) = -sinpsi*xi_p + cospsi*xi_q;
        vTurbPQR(eR) = xi_r;
        

        // vTurbPQR is in the body fixed frame, not NED
        // Rechenschritt wird später in Zeile 158 in FGAuxiliary gemacht : vTurbPQR = in.Tl2b*vTurbPQR;
        

        // hand on the values for the next timestep
        xi_u_km1 = xi_u; nu_u_km1 = nu_u;
        xi_v_km2 = xi_v_km1; xi_v_km1 = xi_v; nu_v_km2 = nu_v_km1; nu_v_km1 = nu_v;
        xi_w_km2 = xi_w_km1; xi_w_km1 = xi_w; nu_w_km2 = nu_w_km1; nu_w_km1 = nu_w;
        xi_p_km1 = xi_p; nu_p_km1 = nu_p;
        xi_q_km1 = xi_q;
        xi_r_km1 = xi_r;

      }
      default: // if nothing match to a turbType
        break; // break = leave the switch environment
      }

  }

  TurbDirection = atan2( vTurbulenceNED(eEast), vTurbulenceNED(eNorth))*radtodeg;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

double FGWinds::CosineGustProfile(double startDuration, double steadyDuration, double endDuration, double elapsedTime)
{
  double factor = 0.0;
  if (elapsedTime >= 0 && elapsedTime <= startDuration) {
    factor = (1.0 - cos(M_PI*elapsedTime/startDuration))/2.0;
  } else if (elapsedTime > startDuration && (elapsedTime <= (startDuration + steadyDuration))) {
    factor = 1.0;
  } else if (elapsedTime > (startDuration + steadyDuration) && elapsedTime <= (startDuration + steadyDuration + endDuration)) {
    factor = (1-cos(M_PI*(1-(elapsedTime-(startDuration + steadyDuration))/endDuration)))/2.0;
  } else {
    factor = 0.0;
  }

  return factor;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void FGWinds::CosineGust()
{
  struct OneMinusCosineProfile& profile = oneMinusCosineGust.gustProfile;

  double factor = CosineGustProfile( profile.startupDuration,
                                     profile.steadyDuration,
                                     profile.endDuration,
                                     profile.elapsedTime);
  // Normalize the gust wind vector
  oneMinusCosineGust.vWind.Normalize();

  if (oneMinusCosineGust.vWindTransformed.Magnitude() == 0.0) {
    switch (oneMinusCosineGust.gustFrame) {
    case gfBody:
      oneMinusCosineGust.vWindTransformed = in.Tl2b.Inverse() * oneMinusCosineGust.vWind;
      break;
    case gfWind:
      oneMinusCosineGust.vWindTransformed = in.Tl2b.Inverse() * in.Tw2b * oneMinusCosineGust.vWind;
      break;
    case gfLocal:
      // this is the native frame - and the default.
      oneMinusCosineGust.vWindTransformed = oneMinusCosineGust.vWind;
      break;
    default:
      break;
    }
  }

  vCosineGust = factor * oneMinusCosineGust.vWindTransformed * oneMinusCosineGust.magnitude;

  profile.elapsedTime += in.totalDeltaT;

  if (profile.elapsedTime > (profile.startupDuration + profile.steadyDuration + profile.endDuration)) {
    profile.Running = false;
    profile.elapsedTime = 0.0;
    oneMinusCosineGust.vWindTransformed.InitMatrix(0.0);
    vCosineGust.InitMatrix(0);
  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void FGWinds::NumberOfUpDownburstCells(int num)
{
  for (unsigned int i=0; i<UpDownBurstCells.size();i++) delete UpDownBurstCells[i];
  UpDownBurstCells.clear();
  if (num >= 0) {
    for (int i=0; i<num; i++) UpDownBurstCells.push_back(new struct UpDownBurst);
  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Calculates the distance between a specified point (where presumably the
// Up/Downburst is centered) and the current vehicle location. The distance
// here is calculated from the Haversine formula.

double FGWinds::DistanceFromRingCenter(double lat, double lon)
{
  double deltaLat = in.latitude - lat;
  double deltaLong = in.longitude - lon;
  double dLat2 = deltaLat/2.0;
  double dLong2 = deltaLong/2.0;
  double a = sin(dLat2)*sin(dLat2)
             + cos(lat)*cos(in.latitude)*sin(dLong2)*sin(dLong2);
  double c = 2.0*atan2(sqrt(a), sqrt(1.0 - a));
  double d = in.planetRadius*c;
  return d;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void FGWinds::UpDownBurst()
{

  for (unsigned int i=0; i<UpDownBurstCells.size(); i++) {
    /*double d =*/ DistanceFromRingCenter(UpDownBurstCells[i]->ringLatitude, UpDownBurstCells[i]->ringLongitude);

  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void FGWinds::bind(void)
{
  typedef double (FGWinds::*PMF)(int) const;
  typedef int (FGWinds::*PMFt)(void) const;
  typedef void   (FGWinds::*PMFd)(int,double);
  typedef void   (FGWinds::*PMFi)(int);
  typedef double (FGWinds::*Ptr)(void) const;

  // User-specified steady, constant, wind properties (local navigational/geographic frame: N-E-D)
  PropertyManager->Tie("atmosphere/psiw-rad", this, &FGWinds::GetWindPsi, &FGWinds::SetWindPsi);
  PropertyManager->Tie("atmosphere/wind-north-fps", this, eNorth, (PMF)&FGWinds::GetWindNED,
                                                          (PMFd)&FGWinds::SetWindNED);
  PropertyManager->Tie("atmosphere/wind-east-fps",  this, eEast, (PMF)&FGWinds::GetWindNED,
                                                          (PMFd)&FGWinds::SetWindNED);
  PropertyManager->Tie("atmosphere/wind-down-fps",  this, eDown, (PMF)&FGWinds::GetWindNED,
                                                          (PMFd)&FGWinds::SetWindNED);
  
  // User-specified mean wind (body frame)
  PropertyManager->Tie("atmosphere/wind-xbody-fps", this, 1, (PMF)&FGWinds::GetWindBody,
                                                          (PMFd)&FGWinds::SetWindBody);
  PropertyManager->Tie("atmosphere/wind-ybody-fps", this, 2, (PMF)&FGWinds::GetWindBody,
                                                          (PMFd)&FGWinds::SetWindBody);
  PropertyManager->Tie("atmosphere/wind-zbody-fps", this, 3, (PMF)&FGWinds::GetWindBody,
                                                          (PMFd)&FGWinds::SetWindBody);
  
  PropertyManager->Tie("atmosphere/wind-mag-fps", this, &FGWinds::GetWindspeed,
                                                        &FGWinds::SetWindspeed);

  // User-specifieded gust (local navigational/geographic frame: N-E-D)
  PropertyManager->Tie("atmosphere/gust-north-fps", this, eNorth, (PMF)&FGWinds::GetGustNED,
                                                          (PMFd)&FGWinds::SetGustNED);
  PropertyManager->Tie("atmosphere/gust-east-fps",  this, eEast, (PMF)&FGWinds::GetGustNED,
                                                          (PMFd)&FGWinds::SetGustNED);
  PropertyManager->Tie("atmosphere/gust-down-fps",  this, eDown, (PMF)&FGWinds::GetGustNED,
                                                          (PMFd)&FGWinds::SetGustNED);

  // User-specified 1 - cosine gust parameters (in specified frame)
  PropertyManager->Tie("atmosphere/cosine-gust/startup-duration-sec", this, (Ptr)0L, &FGWinds::StartupGustDuration);
  PropertyManager->Tie("atmosphere/cosine-gust/steady-duration-sec", this, (Ptr)0L, &FGWinds::SteadyGustDuration);
  PropertyManager->Tie("atmosphere/cosine-gust/end-duration-sec", this, (Ptr)0L, &FGWinds::EndGustDuration);
  PropertyManager->Tie("atmosphere/cosine-gust/magnitude-ft_sec", this, (Ptr)0L, &FGWinds::GustMagnitude);
  PropertyManager->Tie("atmosphere/cosine-gust/frame", this, (PMFt)0L, (PMFi)&FGWinds::GustFrame);
  PropertyManager->Tie("atmosphere/cosine-gust/X-velocity-ft_sec", this, (Ptr)0L, &FGWinds::GustXComponent);
  PropertyManager->Tie("atmosphere/cosine-gust/Y-velocity-ft_sec", this, (Ptr)0L, &FGWinds::GustYComponent);
  PropertyManager->Tie("atmosphere/cosine-gust/Z-velocity-ft_sec", this, (Ptr)0L, &FGWinds::GustZComponent);
  PropertyManager->Tie("atmosphere/cosine-gust/start", this, (PMFt)0L, (PMFi)&FGWinds::StartGust);

  // User-specified Up- Down-burst parameters
  PropertyManager->Tie("atmosphere/updownburst/number-of-cells", this, (PMFt)0L, &FGWinds::NumberOfUpDownburstCells);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);
//  PropertyManager->Tie("atmosphere/updownburst/", this, (Ptr)0L, &FGWinds::);

  // User-specified turbulence (local navigational/geographic frame: N-E-D)
  PropertyManager->Tie("atmosphere/turb-north-fps", this, eNorth, (PMF)&FGWinds::GetTurbNED,
                                                          (PMFd)&FGWinds::SetTurbNED);
  PropertyManager->Tie("atmosphere/turb-east-fps",  this, eEast, (PMF)&FGWinds::GetTurbNED,
                                                          (PMFd)&FGWinds::SetTurbNED);
  PropertyManager->Tie("atmosphere/turb-down-fps",  this, eDown, (PMF)&FGWinds::GetTurbNED,
                                                          (PMFd)&FGWinds::SetTurbNED);
  
    // User-specified turbulence (body frame)
  PropertyManager->Tie("atmosphere/turb-xbody-fps", this, 1, (PMF)&FGWinds::GetTurbBody,
                                                          (PMFd)&FGWinds::SetTurbBody);
  PropertyManager->Tie("atmosphere/turb-ybody-fps", this, 2, (PMF)&FGWinds::GetTurbBody,
                                                          (PMFd)&FGWinds::SetTurbBody);
  PropertyManager->Tie("atmosphere/turb-zbody-fps", this, 3, (PMF)&FGWinds::GetTurbBody,
                                                          (PMFd)&FGWinds::SetTurbBody);
  
  /* Experimental turbulence parameters. Explenation of Tie function under 
  FGPropertyManager.h: Tie a property to a pair of indexed object methods.*/
  PropertyManager->Tie("atmosphere/p-turb-rad_sec", this,1, (PMF)&FGWinds::GetTurbPQR,
                                                            (PMFd)&FGWinds::SetTurbPQR);
  PropertyManager->Tie("atmosphere/q-turb-rad_sec", this,2, (PMF)&FGWinds::GetTurbPQR,                                        
                                                            (PMFd)&FGWinds::SetTurbPQR);
  PropertyManager->Tie("atmosphere/r-turb-rad_sec", this,3, (PMF)&FGWinds::GetTurbPQR,
                                                            (PMFd)&FGWinds::SetTurbPQR);    
  
  PropertyManager->Tie("atmosphere/turb-type", this, (PMFt)&FGWinds::GetTurbType, (PMFi)&FGWinds::SetTurbType);
//  PropertyManager->Tie("atmosphere/wind-model", this, (PMFt)&FGWinds::GetWindModel, (PMFi)&FGWinds::SetWindModel);
  PropertyManager->Tie("atmosphere/turb-rate", this, &FGWinds::GetTurbRate, &FGWinds::SetTurbRate);
  PropertyManager->Tie("atmosphere/turb-gain", this, &FGWinds::GetTurbGain, &FGWinds::SetTurbGain);
  PropertyManager->Tie("atmosphere/turb-rhythmicity", this, &FGWinds::GetRhythmicity,
                                                            &FGWinds::SetRhythmicity);

  // Parameters for milspec turbulence
  PropertyManager->Tie("atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps",
                       this, &FGWinds::GetWindspeed20ft,
                             &FGWinds::SetWindspeed20ft);
  PropertyManager->Tie("atmosphere/turbulence/milspec/severity",
                       this, &FGWinds::GetProbabilityOfExceedence,
                             &FGWinds::SetProbabilityOfExceedence);

  // Total, calculated winds (local navigational/geographic frame: N-E-D). Read only.
  PropertyManager->Tie("atmosphere/total-wind-north-fps", this, eNorth, (PMF)&FGWinds::GetTotalWindNED);
  PropertyManager->Tie("atmosphere/total-wind-east-fps",  this, eEast,  (PMF)&FGWinds::GetTotalWindNED);
  PropertyManager->Tie("atmosphere/total-wind-down-fps",  this, eDown,  (PMF)&FGWinds::GetTotalWindNED);
  
  // Total, calculated winds (body frame)
  PropertyManager->Tie("atmosphere/total-wind-xbody-fps", this, 1, (PMF)&FGWinds::GetTotalWindBody,
                                                          (PMFd)&FGWinds::SetTotalWindBody);
  PropertyManager->Tie("atmosphere/total-wind-ybody-fps", this, 2, (PMF)&FGWinds::GetTotalWindBody,
                                                          (PMFd)&FGWinds::SetTotalWindBody);
  PropertyManager->Tie("atmosphere/total-wind-zbody-fps", this, 3, (PMF)&FGWinds::GetTotalWindBody,
                                                          (PMFd)&FGWinds::SetTotalWindBody);

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//    The bitmasked value choices are as follows:
//    unset: In this case (the default) JSBSim would only print
//       out the normally expected messages, essentially echoing
//       the config files as they are read. If the environment
//       iable is not set, debug_lvl is set to 1 internally
//    0: This requests JSBSim not to output any messages
//       whatsoever.
//    1: This value explicity requests the normal JSBSim
//       startup messages
//    2: This value asks for a message to be printed out when
//       a class is instantiated
//    4: When this value is set, a message is displayed when a
//       FGModel object executes its Run() method
//    8: When this value is set, ious runtime state iables
//       are printed out periodically
//    16: When set ious parameters are sanity checked and
//       a message is printed out when they go out of bounds

void FGWinds::Debug(int from)
{
  if (debug_lvl <= 0) return;

  if (debug_lvl & 1) { // Standard console startup message output
    if (from == 0) { // Constructor
    }
  }
  if (debug_lvl & 2 ) { // Instantiation/Destruction notification
    if (from == 0) cout << "Instantiated: FGWinds" << endl;
    if (from == 1) cout << "Destroyed:    FGWinds" << endl;
  }
  if (debug_lvl & 4 ) { // Run() method entry print for FGModel-derived objects
  }
  if (debug_lvl & 8 ) { // Runtime state iables
  }
  if (debug_lvl & 16) { // Sanity checking
  }
  if (debug_lvl & 128) { //
  }
  if (debug_lvl & 64) {
    if (from == 0) { // Constructor
      cout << IdSrc << endl;
      cout << IdHdr << endl;
    }
  }
}

} // namespace JSBSim
