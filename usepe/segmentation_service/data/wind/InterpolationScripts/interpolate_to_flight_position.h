/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 Header:       FGWinds.h
 Author:       Jon Berndt, Andreas Gaeb, David Culp
 Date started: 5/2011

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

HISTORY
--------------------------------------------------------------------------------
5/2011   JSB   Created

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SENTRY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#ifndef FGWINDS_H
#define FGWINDS_H

// Homogeneous Case.
//#define NX 301
//#define NY 301
//#define NZ 62
//#define NT 4

// Heterogeneous Case.
#define NX 601
#define NY 201
#define NZ 62
#define NT 2

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INCLUDES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#include "models/FGModel.h"
#include "math/FGColumnVector3.h"
#include "math/FGMatrix33.h"
#include "math/FGTable.h"

#include <string>


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEFINITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#define ID_WINDS "$Id: FGWinds.h,v 1.11 2015/02/27 20:36:47 bcoconni Exp $"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FORWARD DECLARATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

namespace JSBSim {

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CLASS DOCUMENTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/** Models atmospheric disturbances: winds, gusts, turbulence, downbursts, etc.

    Various turbulence models are available. They are specified
    via the property <tt>atmosphere/turb-type</tt>. The following models are
    available:
    - 0: ttNone (turbulence disabled)
    - 1: ttStandard
    - 2: ttCulp
    - 3: ttMilspec (Dryden spectrum)
    - 4: ttTustin (Dryden spectrum)

    The Milspec and Tustin models are described in the Yeager report cited below.
    They both use a Dryden spectrum model whose parameters (scale lengths and intensities)
    are modelled according to MIL-F-8785C. Parameters are modelled differently
    for altitudes below 1000ft and above 2000ft, for altitudes in between they
    are interpolated linearly.

    The two models differ in the implementation of the transfer functions
    described in the milspec.

    To use one of these two models, set <tt>atmosphere/turb-type</tt> to 4 resp. 5,
    and specify values for <tt>atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps<tt>
    and <tt>atmosphere/turbulence/milspec/severity<tt> (the latter corresponds to
    the probability of exceedence curves from Fig.&nbsp;7 of the milspec, allowable
    range is 0 (disabled) to 7). <tt>atmosphere/psiw-rad</tt> is respected as well;
    note that you have to specify a positive wind magnitude to prevent psiw from
    being reset to zero.

    Reference values (cf. figures 7 and 9 from the milspec):
    <table>
      <tr><td><b>Intensity</b></td>
          <td><b><tt>windspeed_at_20ft_AGL-fps</tt></b></td>
          <td><b><tt>severity</tt></b></td></tr>
      <tr><td>light</td>
          <td>25 (15 knots)</td>
          <td>3</td></tr>
      <tr><td>moderate</td>
          <td>50 (30 knots)</td>
          <td>4</td></tr>
      <tr><td>severe</td>
          <td>75 (45 knots)</td>
          <td>6</td></tr>
    </table>

    @see Yeager, Jessie C.: "Implementation and Testing of Turbulence Models for
         the F18-HARV" (<a
         href="http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980028448_1998081596.pdf">
         pdf</a>), NASA CR-1998-206937, 1998

    @see MIL-F-8785C: Military Specification: Flying Qualities of Piloted Aircraft

*/

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CLASS DECLARATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

class FGWinds : public FGModel {
public:

  /// Constructor
  FGWinds(FGFDMExec*);
  /// Destructor
  ~FGWinds();
  /** Runs the winds model; called by the Executive
      Can pass in a value indicating if the executive is directing the simulation to Hold.
      @param Holding if true, the executive has been directed to hold the sim from 
                     advancing time. Some models may ignore this flag, such as the Input
                     model, which may need to be active to listen on a socket for the
                     "Resume" command to be given.
      @return false if no error */
  bool Run(bool Holding);
  bool InitModel(void);
  enum tType {ttNone, ttStandard, ttCulp, ttMilspec, ttTustin} turbType;

  // TOTAL WIND access functions (wind + gust + turbulence)
  
  /// Sets and Retrieves the wind model using by FlightGear  
  virtual void SetWindModel(std::string wm) {windmodel = wm;} 
  virtual std::string GetWindModel() const {return windmodel;}

  /// Sets and Retrieves the weather condition using by FlightGear
  virtual void SetWeatherCondition(std::string wc) {tile_type = wc;}
  virtual std::string GetWeatherCondition() const {return tile_type;}

  /// Sets if local weather (Detailed Weather in menu) is running
  virtual void SetLocalWeatherFlag(bool lw) {local_weather_running = lw;}
    
  /// Retrieves the total wind components in NED frame.
  virtual const FGColumnVector3& GetTotalWindNED(void) const { return vTotalWindNED; }
  
  /// Retrieves a total wind component in NED frame.
  virtual double GetTotalWindNED(int idx) const {return vTotalWindNED(idx);}

  /// Retrieves a total wind component in body frame.
  virtual double GetTotalWindBody(int idx) const {return vTotalWindBody(idx);}
  
  // WIND access functions

  /// Sets a total wind component in body frame
  virtual void SetTotalWindBody(int idx, double total) { vTotalWindBody(idx)=total;}  
  
  /// Sets the wind components in NED frame.
  virtual void SetWindNED(double wN, double wE, double wD) { vWindNED(1)=wN; vWindNED(2)=wE; vWindNED(3)=wD;}

  /// Sets a wind component in NED frame.
  virtual void SetWindNED(int idx, double wind) { vWindNED(idx)=wind;}
  
  /// Sets a wind component in body frame.
  virtual void SetWindBody(int idx, double wind) { vWindBody(idx)=wind;}

  /// Sets the wind components in NED frame.
  virtual void SetWindNED(const FGColumnVector3& wind) { vWindNED=wind; }

  /// Retrieves the wind components in NED frame.
  virtual const FGColumnVector3& GetWindNED(void) const { return vWindNED; }

  /// Retrieves a wind component in NED frame.
  virtual double GetWindNED(int idx) const {return vWindNED(idx);}

  /// Retrieves a wind component in body frame.
  virtual double GetWindBody(int idx) const {return vWindBody(idx);}
  
  /** Retrieves the direction that the wind is coming from.
      The direction is defined as north=0 and increases counterclockwise.
      The wind heading is returned in radians.*/
  virtual double GetWindPsi(void) const { return psiw; }

  /** Sets the direction that the wind is coming from.
      The direction is defined as north=0 and increases counterclockwise to 2*pi (radians). The
      vertical component of wind is assumed to be zero - and is forcibly set to zero. This function
      sets the vWindNED vector components based on the supplied direction. The magnitude of
      the wind set in the vector is preserved (assuming the vertical component is non-zero).
      @param dir wind direction in the horizontal plane, in radians.*/
  virtual void SetWindPsi(double dir);

  virtual void SetWindspeed(double speed);

  virtual double GetWindspeed(void) const;

  // GUST access functions

  /// Sets a gust component in NED frame.
  virtual void SetGustNED(int idx, double gust) { vGustNED(idx)=gust;}

  /// Sets a turbulence component in NED frame.
  virtual void SetTurbNED(int idx, double turb) { vTurbulenceNED(idx)=turb;}

  /// Sets a turbulence component in body frame.
  virtual void SetTurbBody(int idx, double turb) { vTurbulenceBody(idx)=turb;}

  /// Sets the gust components in NED frame.
  virtual void SetGustNED(double gN, double gE, double gD) { vGustNED(eNorth)=gN; vGustNED(eEast)=gE; vGustNED(eDown)=gD;}

  /// Retrieves a gust component in NED frame.
  virtual double GetGustNED(int idx) const {return vGustNED(idx);}

  /// Retrieves a turbulence component in NED frame.
  virtual double GetTurbNED(int idx) const {return vTurbulenceNED(idx);}

  /// Retrieves a turbulence component in NED frame.
  virtual double GetTurbBody(int idx) const {return vTurbulenceBody(idx);}
  
  /// Retrieves the gust components in NED frame.
  virtual const FGColumnVector3& GetGustNED(void) const {return vGustNED;}

  /** Turbulence models available: ttNone, ttStandard, ttBerndt, ttCulp, ttMilspec, ttTustin */
  virtual void   SetTurbType(tType tt) {turbType = tt;}
  virtual tType  GetTurbType() const {return turbType;}
  
  virtual void   SetTurbGain(double tg) {TurbGain = tg;}
  virtual double GetTurbGain() const {return TurbGain;}

  virtual void   SetTurbRate(double tr) {TurbRate = tr;}
  virtual double GetTurbRate() const {return TurbRate;}

  virtual void   SetRhythmicity(double r) {Rhythmicity=r;}
  virtual double GetRhythmicity() const {return Rhythmicity;}

  virtual double GetTurbPQR(int idx) const {return vTurbPQR(idx);}
  
  /// Sets a turbulence component in NED frame.
  virtual void SetTurbPQR(int idx, double turb) { vTurbPQR(idx)=turb;}
  
  virtual double GetTurbMagnitude(void) const {return vTurbulenceNED.Magnitude();}
  virtual double GetTurbDirection(void) const {return TurbDirection;}
  virtual const FGColumnVector3& GetTurbPQR(void) const {return vTurbPQR;}

  virtual void   SetWindspeed20ft(double ws) { windspeed_at_20ft = ws;}
  virtual double GetWindspeed20ft() const { return windspeed_at_20ft;}

  /// allowable range: 0-7, 3=light, 4=moderate, 6=severe turbulence
  virtual void   SetProbabilityOfExceedence( int idx) {probability_of_exceedence_index = idx;}
  virtual int    GetProbabilityOfExceedence() const { return probability_of_exceedence_index;}

  // Stores data defining a 1 - cosine gust profile that builds up, holds steady
  // and fades out over specified durations.
  struct OneMinusCosineProfile {
    bool Running;           ///<- This flag is set true through FGWinds::StartGust().
    double elapsedTime;     ///<- Stores the elapsed time for the ongoing gust.
    double startupDuration; ///<- Specifies the time it takes for the gust startup transient.
    double steadyDuration;  ///<- Specifies the duration of the steady gust.
    double endDuration;     ///<- Specifies the time it takes for the gust to subsude.
    OneMinusCosineProfile() ///<- The constructor.
    {
      elapsedTime = 0.0;
      Running = false;
      startupDuration = 2;
      steadyDuration = 4;
      endDuration = 2;
    }
  };

  enum eGustFrame {gfNone=0, gfBody, gfWind, gfLocal};

  /// Stores the information about a single one minus cosine gust instance.
  struct OneMinusCosineGust {
    FGColumnVector3 vWind;                    ///<- The input normalized wind vector.
    FGColumnVector3 vWindTransformed;         ///<- The transformed normal vector at the time the gust is started.
    double magnitude;                         ///<- The magnitude of the wind vector.
    eGustFrame gustFrame;                     ///<- The frame that the wind vector is specified in.
    struct OneMinusCosineProfile gustProfile; ///<- The gust shape (profile) data for this gust.
    OneMinusCosineGust()                      ///<- Constructor.
    {
      vWind.InitMatrix(0.0);
      gustFrame = gfLocal;
      magnitude = 1.0;
    };
  };

  /// Stores information about a specified Up- or Down-burst.
  struct UpDownBurst {
    double ringLatitude;                           ///<- The latitude of the downburst run (radians)
    double ringLongitude;                          ///<- The longitude of the downburst run (radians)
    double ringAltitude;                           ///<- The altitude of the ring (feet).
    double ringRadius;                             ///<- The radius of the ring (feet).
    double ringCoreRadius;                         ///<- The cross-section "core" radius of the ring (feet).
    double circulation;                            ///<- The circulation (gamma) (feet-squared per second).
    struct OneMinusCosineProfile oneMCosineProfile;///<- A gust profile structure.
    UpDownBurst() {                                ///<- Constructor
      ringLatitude = ringLongitude = 0.0;
      ringAltitude = 1000.0;
      ringRadius = 2000.0;
      ringCoreRadius = 100.0;
      circulation = 100000.0;
    }
  };

  // 1 - Cosine gust setters
  /// Initiates the execution of the gust.
  virtual void StartGust(bool running) {oneMinusCosineGust.gustProfile.Running = running;}
  ///Specifies the duration of the startup portion of the gust.
  virtual void StartupGustDuration(double dur) {oneMinusCosineGust.gustProfile.startupDuration = dur;}
  ///Specifies the length of time that the gust is at a steady, full strength.
  virtual void SteadyGustDuration(double dur) {oneMinusCosineGust.gustProfile.steadyDuration = dur;}
  /// Specifies the length of time it takes for the gust to return to zero velocity.
  virtual void EndGustDuration(double dur) {oneMinusCosineGust.gustProfile.endDuration = dur;}
  /// Specifies the magnitude of the gust in feet/second.
  virtual void GustMagnitude(double mag) {oneMinusCosineGust.magnitude = mag;}
  /** Specifies the frame that the gust direction vector components are specified in. The 
      body frame is defined with the X direction forward, and the Y direction positive out
      the right wing. The wind frame is defined with the X axis pointing into the velocity
      vector, the Z axis perpendicular to the X axis, in the aircraft XZ plane, and the Y
      axis completing the system. The local axis is a navigational frame with X pointing north,
      Y pointing east, and Z pointing down. This is a locally vertical, locally horizontal
      frame, with the XY plane tangent to the geocentric surface. */
  virtual void GustFrame(eGustFrame gFrame) {oneMinusCosineGust.gustFrame = gFrame;}
  /// Specifies the X component of velocity in the specified gust frame (ft/sec).
  virtual void GustXComponent(double x) {oneMinusCosineGust.vWind(eX) = x;}
  /// Specifies the Y component of velocity in the specified gust frame (ft/sec).
  virtual void GustYComponent(double y) {oneMinusCosineGust.vWind(eY) = y;}
  /// Specifies the Z component of velocity in the specified gust frame (ft/sec).
  virtual void GustZComponent(double z) {oneMinusCosineGust.vWind(eZ) = z;}

  // Up- Down-burst functions
  void NumberOfUpDownburstCells(int num);

  struct Inputs {
    double V;
    double wingspan;
    double DistanceAGL;
    double AltitudeASL;
    double longitude;
    double latitude;
    double planetRadius;
    FGMatrix33 Tl2b;
    FGMatrix33 Tb2l;
    FGMatrix33 Tw2b;
    double totalDeltaT;
  } in;

private:
//For turbulence calculation
  
  // added code by myself
  std::string windmodel; // You have to set #include <string>
  std::string tile_type;
  bool local_weather_running;
  bool input_PALM_data_flag;
  double U;
  double U_rotated;
  double V;
  double V_rotated;
  float u_palm[NZ][NY][NX];
  float u_palm_rotated[NZ][NY][NX];
  float u_av_palm_rotated[NZ][NY][NX];
  float u_av_palm[NZ][NY][NX];
  float v_palm[NZ][NY][NX];
  float v_palm_rotated[NZ][NY][NX];
  float v_av_palm_rotated[NZ][NY][NX];
  float v_av_palm[NZ][NY][NX];  
  float w_palm[NZ][NY][NX];
  float w_av_palm[NZ][NY][NX];
  float u_xyav_palm[NZ];
  float v_xyav_palm[NZ];
  float delta_x;
  float dx;
  float dy;
  float delta_y;
  float delta_z;
  float xsize;
  float ysize;
  float zsize;
  float zu_max;
  float xp_old;
  float yp_old;
  float zp_old;
  float xp_new;
  float yp_new;
  float zp_new;
  int i,j,k;
  float help1;
  float help2;
  float help3;
  float help4;
  float help5;
  float help6;
  float xp_new_1;
  float yp_new_1;
  float xp_new_2;
  float yp_new_2;
  float xp_new_3;
  float yp_new_3;
  float FT2M;
  float M2FT;
  float KT2MPS;
  float MPS2KT;
  float vtailarm;
  float u_windspeed;
  float u_windspeed_1;
  float u_windspeed_2;
  float u_windspeed_1_WF;
  float u_windspeed_2_WF;  
  float u_windspeed_3;
  float u_xyav_windspeed_current;
  float u_xyav_palm_20ft;
  float u_squared;
  float u_top;
  float v_xyav_windspeed_current;
  float v_windspeed;
  float v_windspeed_WF;
  float v_windspeed_3;
  float v_windspeed_1;
  float v_windspeed_2;
  float v_windspeed_3_WF;
  float v_xyav_palm_20ft;
  float v_squared;
  float v_top;
  float w_windspeed;
  float w_windspeed_1;
  float w_windspeed_2;
  float w_windspeed_3;
  float w_xyav_windspeed_current;
  float windspeed_ground;
  float winddir_ground;
  float windspeed_top;
  float winddir_top;
  float winddir;
  float windspeed_current_ms;
  float windspeed_current_kt;
  float windspeed_current_fps;
  float delta_t;
  float V_g;
  float phi_heading;
  float elapsed_time;
  float r1g;
  float r2g;
  float wind_from_north;
  float wind_from_east;
  float wind_from_down;
  float x_distance;
  float y_distance;
  float distance_h;
  float alpha;
  float rotationangle;
  FGColumnVector3 windspeed_body;
  FGColumnVector3 windspeed_1_body;
  FGColumnVector3 windspeed_2_body;
  FGColumnVector3 windspeed_3_body;
  FGColumnVector3 windspeed_NED;
  FGColumnVector3 windspeed_1_NED;
  FGColumnVector3 windspeed_2_NED;
  FGColumnVector3 windspeed_3_NED;
  SGPropertyNode* Getnode;
  SGPropertyNode* Setnode;
  
  double MagnitudedAccelDt, MagnitudeAccel, Magnitude, TurbDirection;
  //double h;
  double TurbGain;
  double TurbRate;
  double Rhythmicity;
  double wind_from_clockwise;
  double spike, target_time, strength;
  FGColumnVector3 vTurbulenceGrad;
  FGColumnVector3 vBodyTurbGrad;
  FGColumnVector3 vTurbPQR;

  struct OneMinusCosineGust oneMinusCosineGust;
  std::vector <struct UpDownBurst*> UpDownBurstCells;

  // Dryden turbulence model
  double windspeed_at_20ft; ///< in ft/s
  int probability_of_exceedence_index; ///< this is bound as the severity property
  FGTable *POE_Table; ///< probability of exceedence table

  double psiw;
  FGColumnVector3 vTotalWindNED;
  FGColumnVector3 vTotalWindBody;
  FGColumnVector3 vWindNED;
  FGColumnVector3 vWindBody;
  FGColumnVector3 vGustNED;
  FGColumnVector3 vCosineGust;
  FGColumnVector3 vBurstGust;
  FGColumnVector3 vTurbulenceNED;
  FGColumnVector3 vTurbulenceBody;
  
  void Turbulence(double h);
  void PALMLoop();
  int InputPALM(void);
  void UpDownBurst();

  void CosineGust();
  double CosineGustProfile( double startDuration, double steadyDuration,
                            double endDuration, double elapsedTime);
  double DistanceFromRingCenter(double lat, double lon);

  virtual void bind(void);
  void Debug(int from);
};

} // namespace JSBSim

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#endif

