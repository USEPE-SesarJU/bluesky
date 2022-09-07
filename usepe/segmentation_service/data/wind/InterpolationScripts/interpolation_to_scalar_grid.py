# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:58:20 2021

@author: Giers
"""


### Start script ###



### import module ###
import netCDF4 as nc4      #for reading NetCDF files
import matplotlib.pyplot as plt  #for plotting
import matplotlib.patches as patches #for patches
import matplotlib.colors as col #colormap
import numpy as np



### Import PALM files with the intension of reading it ###
#parent domain with 10m resolution
f1 = nc4.Dataset("/localdata/giersch/AG_Raasch/DroneWaves/DW_convective_3ms_nesting/"
"OUTPUT/DW_convective_3ms_nesting_av_3d.001.nc",'r')

#first child domain in the north which contains the vertiports (2m resolution)
f2 = nc4.Dataset("/localdata/giersch/AG_Raasch/DroneWaves/DW_convective_3ms_nesting/"
"OUTPUT/DW_convective_3ms_nesting_av_3d_N02.001.nc",'r')

#second child domain in the south which contains the heliport (2m resolution)
f3 = nc4.Dataset("/localdata/giersch/AG_Raasch/DroneWaves/DW_convective_3ms_nesting/"
"OUTPUT/DW_convective_3ms_nesting_av_3d_N03.001.nc",'r')



### Define the input data to be analyzed. Maximum building height ~500m        ###
time_index1 = 0   # time index
k1          = 56   # height index k

time_index2 = 0   # time index
k2          = 276  # height index k

time_index3 = 0   # time index
k3          = 276  # height index k



### Accessing PALM data from a variable and directly load it into memory. This ###
### is necessary if further index calculations are carried out but the reading ###
### to memory needs more time. For get a quick access on disk level use e.g.   ###
### u = f.variables['u']. See also                                             ###
### http://xarray.pydata.org/en/stable/io.html#reading-and-writing-files       ### 
#file 1                                           
time1 = f1.variables['time'][:]

x1 = f1.variables['x'][:]

y1 = f1.variables['y'][:]

zu_3d1 = f1.variables['zu_3d'][:]

u1 = f1.variables['u'][time_index1,:k1,:,:]

v1 = f1.variables['v'][time_index1,:k1,:,:]

w1 = f1.variables['w'][time_index1,:k1,:,:]

u21 = f1.variables['u2'][time_index1,:k1,:,:]

v21 = f1.variables['v2'][time_index1,:k1,:,:]

w21 = f1.variables['w2'][time_index1,:k1,:,:]

zusi1 = f1.variables['zusi'][:,:] # height of topography top on scalar grid. In PALM code zu_s_inner

zwwi1 = f1.variables['zwwi'][:,:] # height of topography top on w grid. In PALM code zw_w_inner



#file 2
time2 = f2.variables['time'][:]

x2 = f2.variables['x'][:]

y2 = f2.variables['y'][:]

zu_3d2 = f2.variables['zu_3d'][:]

u2 = f2.variables['u'][time_index2,:k2,:,:]

v2 = f2.variables['v'][time_index2,:k2,:,:]

w2 = f2.variables['w'][time_index2,:k2,:,:]

u22 = f2.variables['u2'][time_index2,:k2,:,:]

v22 = f2.variables['v2'][time_index2,:k2,:,:]

w22 = f2.variables['w2'][time_index2,:k2,:,:]

zusi2 = f2.variables['zusi'][:,:] # height of topography top on scalar grid. In PALM code zu_s_inner

zwwi2 = f2.variables['zwwi'][:,:] # height of topography top on w grid. In PALM code zw_w_inner



#file 3
time3 = f3.variables['time'][:]

x3 = f3.variables['x'][:]

y3 = f3.variables['y'][:]

zu_3d3 = f3.variables['zu_3d'][:]

u3 = f3.variables['u'][time_index3,:k3,:,:]

v3 = f3.variables['v'][time_index3,:k3,:,:]

w3 = f3.variables['w'][time_index3,:k3,:,:]

u23 = f3.variables['u2'][time_index3,:k3,:,:]

v23 = f3.variables['v2'][time_index3,:k3,:,:]

w23 = f3.variables['w2'][time_index3,:k3,:,:]

zusi3 = f3.variables['zusi'][:,:] # height of topography top on scalar grid. In PALM code zu_s_inner

zwwi3 = f3.variables['zwwi'][:,:] # height of topography top on w grid. In PALM code zw_w_inner



### Close files ###
f1.close()
f2.close()
f3.close()



### Get dimensions ###
dim1 = u1.shape
dim2 = u2.shape
dim3 = u3.shape


# Assign shape (tuple) to individual dimensions (int)
dimz1 = dim1[0]
dimy1 = dim1[1]
dimx1 = dim1[2]

dimz2 = dim2[0]
dimy2 = dim2[1]
dimx2 = dim2[2]

dimz3 = dim3[0]
dimy3 = dim3[1]
dimx3 = dim3[2]



### Declaration and Initialization of variables and arrays that will be used ###
u_int1      = np.ndarray((dimz1,dimy1,dimx1),dtype=float,order="C")
u_int1_plot = np.ndarray((dimy1,dimx1),dtype=float,order="C")
v_int1      = np.ndarray((dimz1,dimy1,dimx1),dtype=float,order="C")
v_int1_plot = np.ndarray((dimy1,dimx1),dtype=float,order="C")
tke1        = np.ndarray((dimz1,dimy1,dimx1),dtype=float,order="C")
tke1_plot   = np.ndarray((dimy1,dimx1),dtype=float,order="C")
u_var_int1  = np.ndarray((dimz1,dimy1,dimx1),dtype=float,order="C")
v_var_int1  = np.ndarray((dimz1,dimy1,dimx1),dtype=float,order="C")
w_var_int1  = np.ndarray((dimz1,dimy1,dimx1),dtype=float,order="C")
height_index_zu1 = np.ndarray((dimy1,dimx1),dtype=int,order="C")
height_index_zw1 = np.ndarray((dimy1,dimx1),dtype=int,order="C")

u_int2      = np.ndarray((dimz2,dimy2,dimx2),dtype=float,order="C")
u_int2_plot = np.ndarray((dimy2,dimx2),dtype=float,order="C")
v_int2      = np.ndarray((dimz2,dimy2,dimx2),dtype=float,order="C")
v_int2_plot = np.ndarray((dimy2,dimx2),dtype=float,order="C")
tke2        = np.ndarray((dimz2,dimy2,dimx2),dtype=float,order="C")
tke2_plot   = np.ndarray((dimy2,dimx2),dtype=float,order="C")
u_var_int2  = np.ndarray((dimz2,dimy2,dimx2),dtype=float,order="C")
v_var_int2  = np.ndarray((dimz2,dimy2,dimx2),dtype=float,order="C")
w_var_int2  = np.ndarray((dimz2,dimy2,dimx2),dtype=float,order="C")
height_index_zu2 = np.ndarray((dimy2,dimx2),dtype=int,order="C")
height_index_zw2 = np.ndarray((dimy2,dimx2),dtype=int,order="C")

u_int3      = np.ndarray((dimz3,dimy3,dimx3),dtype=float,order="C")
u_int3_plot = np.ndarray((dimy3,dimx3),dtype=float,order="C")
v_int3      = np.ndarray((dimz3,dimy3,dimx3),dtype=float,order="C")
v_int3_plot = np.ndarray((dimy3,dimx3),dtype=float,order="C")
tke3        = np.ndarray((dimz3,dimy3,dimx3),dtype=float,order="C")
tke3_plot   = np.ndarray((dimy3,dimx3),dtype=float,order="C")
u_var_int3  = np.ndarray((dimz3,dimy3,dimx3),dtype=float,order="C")
v_var_int3  = np.ndarray((dimz3,dimy3,dimx3),dtype=float,order="C")
w_var_int3  = np.ndarray((dimz3,dimy3,dimx3),dtype=float,order="C")
height_index_zu3 = np.ndarray((dimy3,dimx3),dtype=int,order="C")
height_index_zw3 = np.ndarray((dimy3,dimx3),dtype=int,order="C")


u_int1[:,:,:]     = -9999.0
u_int1_plot[:,:]  = -9999.0
v_int1[:,:,:]     = -9999.0
v_int1_plot[:,:]  = -9999.0
tke1[:,:,:]       = -9999.0
tke1_plot[:,:]  = -9999.0
u_var_int1[:,:,:] = -9999.0
v_var_int1[:,:,:] = -9999.0
w_var_int1[:,:,:] = -9999.0

u_int2[:,:,:]     = -9999.0
u_int2_plot[:,:]  = -9999.0
v_int2[:,:,:]     = -9999.0
v_int2_plot[:,:]  = -9999.0
tke2[:,:,:]       = -9999.0
tke2_plot[:,:]  = -9999.0
u_var_int2[:,:,:] = -9999.0
v_var_int2[:,:,:] = -9999.0
w_var_int2[:,:,:] = -9999.0

u_int3[:,:,:]     = -9999.0
u_int3_plot[:,:]  = -9999.0
v_int3[:,:,:]     = -9999.0
v_int3_plot[:,:]  = -9999.0
tke3[:,:,:]       = -9999.0
tke3_plot[:,:]  = -9999.0
u_var_int3[:,:,:] = -9999.0
v_var_int3[:,:,:] = -9999.0
w_var_int3[:,:,:] = -9999.0



### Fill the masked array elements with zero ###
u1       = u1.filled(0)
v1       = v1.filled(0)
w1       = w1.filled(0)
u21      = u21.filled(0)
v21      = v21.filled(0)
w21      = w21.filled(0)

u2       = u2.filled(0)
v2       = v2.filled(0)
w2       = w2.filled(0)
u22      = u22.filled(0)
v22      = v22.filled(0)
w22      = w22.filled(0)

u3       = u3.filled(0)
v3       = v3.filled(0)
w3       = w3.filled(0)
u23      = u23.filled(0)
v23      = v23.filled(0)
w23      = w23.filled(0)



### Caculate gird spacings ###
dz1 = zu_3d1[2]-zu_3d1[1]
dz2 = zu_3d2[2]-zu_3d2[1]
dz3 = zu_3d3[2]-zu_3d3[1]



### Calculate time-averaged variances via temporal EC-method ###
u_var1      = u21 - u1**2.0
v_var1      = v21 - v1**2.0
w_var1      = w21 - w1**2.0

u_var2      = u22 - u2**2.0
v_var2      = v22 - v2**2.0
w_var2      = w22 - w2**2.0

u_var3      = u23 - u3**2.0
v_var3      = v23 - v3**2.0
w_var3      = w23 - w3**2.0



### Interpolation of velocity components and variances to scalar grid x,y,zu   ###
# Interpolate w-component to zu-grid
w_var_int1[0,:,:]  = w_var1[0,:,:]

k = 1
while (k < dimz1):
    w_var_int1[k,:,:]  = (w_var1[k,:,:] + w_var1[k-1,:,:]) * 0.5
    k = k + 1
    
k = 1
while (k < dimz2):
    w_var_int2[k,:,:]  = (w_var2[k,:,:] + w_var2[k-1,:,:]) * 0.5
    k = k + 1
    
k = 1
while (k < dimz3):
    w_var_int3[k,:,:]  = (w_var3[k,:,:] + w_var3[k-1,:,:]) * 0.5
    k = k + 1



# Interpolate u-component to x-gird
i = 0
while (i < dimx1-1):
    u_int1[:,:,i]      = (u1[:,:,i+1] + u1[:,:,i]) * 0.5
    u_var_int1[:,:,i]  = (u_var1[:,:,i+1] + u_var1[:,:,i]) * 0.5
    
    i = i + 1
    
i = 0
while (i < dimx2-1):
    u_int2[:,:,i]      = (u2[:,:,i+1] + u2[:,:,i]) * 0.5
    u_var_int2[:,:,i]  = (u_var2[:,:,i+1] + u_var2[:,:,i]) * 0.5
    
    i = i + 1

i = 0
while (i < dimx3-1):
    u_int3[:,:,i]      = (u3[:,:,i+1] + u3[:,:,i]) * 0.5
    u_var_int3[:,:,i]  = (u_var3[:,:,i+1] + u_var3[:,:,i]) * 0.5
    
    i = i + 1
    
u_int1[:,:,dimx1-1]      = u1[:,:,dimx1-1]
u_var_int1[:,:,dimx1-1]  = u_var1[:,:,dimx1-1]
u_int2[:,:,dimx2-1]      = u2[:,:,dimx2-1]
u_var_int2[:,:,dimx2-1]  = u_var2[:,:,dimx2-1]
u_int3[:,:,dimx3-1]      = u3[:,:,dimx3-1]
u_var_int3[:,:,dimx3-1]  = u_var3[:,:,dimx3-1]



# Interpolate v-component to y-grid
j = 0
while (j < dimy1-1):
    v_int1[:,j,:]      = (v1[:,j+1,:] + v1[:,j,:]) * 0.5
    v_var_int1[:,j,:]  = (v_var1[:,j+1,:] + v_var1[:,j,:]) * 0.5

    j = j + 1

j = 0
while (j < dimy2-1):
    v_int2[:,j,:]      = (v2[:,j+1,:] + v2[:,j,:]) * 0.5
    v_var_int2[:,j,:]  = (v_var2[:,j+1,:] + v_var2[:,j,:]) * 0.5
    
    j = j + 1
    
j = 0
while (j < dimy3-1):
    v_int3[:,j,:]      = (v3[:,j+1,:] + v3[:,j,:]) * 0.5
    v_var_int3[:,j,:]  = (v_var3[:,j+1,:] + v_var3[:,j,:]) * 0.5
    
    j = j + 1

v_int1[:,dimy1-1,:]      = v1[:,dimy1-1,:]
v_var_int1[:,dimy1-1,:]  = v_var1[:,dimy1-1,:]
v_int2[:,dimy2-1,:]      = v2[:,dimy2-1,:]
v_var_int2[:,dimy2-1,:]  = v_var2[:,dimy2-1,:]
v_int3[:,dimy3-1,:]      = v3[:,dimy3-1,:]
v_var_int3[:,dimy3-1,:]  = v_var3[:,dimy3-1,:]



### Calculate resolved-scale TKE on scalar grid ###
tke1 = 0.5 * (u_var_int1 + v_var_int1 + w_var_int1)
tke2 = 0.5 * (u_var_int2 + v_var_int2 + w_var_int2)
tke3 = 0.5 * (u_var_int3 + v_var_int3 + w_var_int3)



# Calculate index bulding height and the bulding height following output.
# Above ground level = Mean sea level + Terrain altitude + building height
# Row-major order (C-style) is the default in NumPy, which means that the last
# index runs fastest (a11,a12,a13,...,a21,a22,a23,...; x[rows, columns]).
# https://agilescientific.com/blog/2018/12/28/what-is-the-fastest-axis-of-an-array
i             = 0
j             = 0
k_index_shift = 1  # height index to plot (5m above ground. The Building ends at the zw-grid.)

while j < dimy1:
    while i < dimx1:
        if (zusi1[j,i] > 0):

            height_index_zu1[j,i] = (zusi1[j,i]+dz1*0.5)/dz1
            height_index_zw1[j,i] = zwwi1[j,i]/dz1
            tke1_plot[j,i] = tke1[height_index_zu1[j,i]+k_index_shift,j,i]

        elif (zusi1[j,i] == 0):
            height_index_zu1[j,i] = 0
            height_index_zw1[j,i] = 0
            tke1_plot[j,i] = -9999.0
            
        u_int1_plot[j,i] = u_int1[height_index_zu1[j,i]+k_index_shift,j,i]
        v_int1_plot[j,i] = v_int1[height_index_zu1[j,i]+k_index_shift,j,i]
        i = i+1

    i = 0
    j = j+1

i             = 0
j             = 0
k_index_shift = 3  # height index to plot (5m above ground). The building ends at the zw-grid)

while j < dimy2:
    while i < dimx2:
        if (zusi2[j,i] > 0):

            height_index_zu2[j,i] = (zusi2[j,i]+dz2*0.5)/dz2
            height_index_zw2[j,i] = zwwi2[j,i]/dz2
            tke2_plot[j,i] = tke2[height_index_zu2[j,i]+k_index_shift,j,i]

        elif (zusi2[j,i] == 0):
            height_index_zu2[j,i] = 0
            height_index_zw2[j,i] = 0
            tke2_plot[j,i] = -9999.0

        u_int2_plot[j,i] = u_int2[height_index_zu2[j,i]+k_index_shift,j,i]
        v_int2_plot[j,i] = v_int2[height_index_zu2[j,i]+k_index_shift,j,i]
        i = i+1

    i = 0
    j = j+1

i             = 0
j             = 0
k_index_shift = 3  # height index to plot (5m above ground). The building ends at the zw-grid)

while j < dimy3:
    while i < dimx3:
        if (zusi3[j,i] > 0):

            height_index_zu3[j,i] = (zusi3[j,i]+dz3*0.5)/dz3
            height_index_zw3[j,i] = zwwi3[j,i]/dz3
            tke3_plot[j,i] = tke3[height_index_zu3[j,i]+k_index_shift,j,i]

        elif (zusi3[j,i] == 0):
            height_index_zu3[j,i] = 0
            height_index_zw3[j,i] = 0
            tke3_plot[j,i] = -9999.0

        u_int3_plot[j,i] = u_int3[height_index_zu3[j,i]+k_index_shift,j,i]
        v_int3_plot[j,i] = v_int3[height_index_zu3[j,i]+k_index_shift,j,i]
        i = i+1

    i = 0
    j = j+1
  
print(zusi1[283,276])
print(zwwi1[283,276])
print(height_index_zu1[283,276]) 
print(height_index_zw1[283,276])
print(zusi3[252,275])
print(zwwi3[252,275])
print(height_index_zu3[252,275]) 
print(height_index_zw3[252,275])    

### Create the figure ###
# General settings. The values for figsize represents the pixels since dpi*inch=px
# figsize width, height in inches.
my_dpi = 300
fig = plt.figure(num=1, figsize=(3000/my_dpi, 2700/my_dpi), dpi=my_dpi)
fig.suptitle('$z$ = 5$\,$m above ground', fontsize=10, x=0.125, y=0.9, ha='left')

#Settings for arrows
arrow_fc='white'
arrow_ec='black'



# Start first subfigure
ax1 = fig.add_subplot(1,2,1) 

# x-axis settings
ax1.set_xlabel("$x$ [m]", fontsize=12)
ax1.set_xlim(x1[0],x1[dimx1-1])

# y-axis settings
ax1.set_ylabel("$y$ [m]", fontsize=12)
ax1.set_ylim(y1[0],y1[dimy1-1])

# title
ax1.set_title("a) parent", loc='left', y=-0.1)

# set background color of subplot
ax1.set_facecolor('white')

con = ax1.contourf(x1,y1,tke1_plot,np.arange(0.0,1.51,0.15),extend='max',
                   cmap='jet')
#arrow1 = ax1.quiver(x1[::18],y1[::18],u_int1_plot[::18,::18],v_int1_plot[::18,::18],
#                   facecolor=arrow_fc,edgecolor=arrow_ec, linewidth=1,
#                   scale_units='inches',scale=8, width=0.008,
#                   headwidth=2.5, headlength=3, headaxislength=2.5)
#qk = ax1.quiverkey(arrow1, 0.76, 0.57, 2, r'2$\,$m$\,$s$^{-1}$', labelpos='E',
#                   coordinates='figure', facecolor=arrow_fc,edgecolor=arrow_ec,
#                   labelcolor='black')
ax1.streamplot(x1, y1, u_int1_plot, v_int1_plot, density=2, 
               color='grey',linewidth=0.5)

# Add child domain boundaries
rect1 = patches.Rectangle((1600, 5300), 1040, 1040, linewidth=3, edgecolor='black', facecolor='none')
rect2 = patches.Rectangle((2200, 2320), 1040, 1040, linewidth=3, edgecolor='black', facecolor='none')
ax1.add_patch(rect1)
ax1.add_patch(rect2)



# Start second subfigure
ax2 = fig.add_subplot(100,100,(60,4389)) 

# x-axis settings
ax2.set_xlabel("$x$ [m]", fontsize=12)
ax2.set_xlim(x2[0],x2[dimx2-1])

# y-axis settings
ax2.set_ylabel("$y$ [m]", fontsize=12)
ax2.set_ylim(y2[0],y2[dimy2-1])

# title
ax2.set_title("b) child 1", loc='left', y=-0.225)

# set background color of subplot
ax2.set_facecolor('white')

con = ax2.contourf(x2,y2,tke2_plot,np.arange(0.0,1.51,0.15),extend='max',
                   cmap='jet')
#arrow2 = ax2.quiver(x2[::18],y2[::18],u_int2_plot[::18,::18],v_int2_plot[::18,::18],
#                   facecolor=arrow_fc,edgecolor=arrow_ec, linewidth=1,
#                   scale_units='inches',scale=8, width=0.01,
#                   headwidth=2.5, headlength=3, headaxislength=2.5)
ax2.streamplot(x2, y2, u_int2_plot, v_int2_plot, density=1.5, 
               color='grey',linewidth=0.5)

# Add vertiports
vertiports_x_loc = [221,251,241]
vertiports_y_loc = [221,351,461]
ax2.plot(vertiports_x_loc, vertiports_y_loc, markerfacecolor='black', marker='o', markersize=6,
         markeredgecolor='black', linestyle='none')



# Start third subfigure
ax3 = fig.add_subplot(100,100,(5660,9989) )

# x-axis settings
ax3.set_xlabel("$x$ [m]", fontsize=12)
ax3.set_xlim(x3[0],x3[dimx3-1])

# y-axis settings
ax3.set_ylabel("$y$ [m]", fontsize=12)
ax3.set_ylim(y3[0],y3[dimy3-1])

# title
ax3.set_title("c) child 2", loc='left', y=-0.225)

# set background color of subplot
ax3.set_facecolor('white')

con = ax3.contourf(x3,y3,tke3_plot,np.arange(0.0,1.51,0.15),extend='max',
                   cmap='jet')
#arrow3 = ax3.quiver(x3[::20],y3[::20],u_int3_plot[::20,::20],v_int3_plot[::20,::20],
#                    facecolor=arrow_fc,edgecolor=arrow_ec, linewidth=1,
#                    scale_units='inches',scale=8, width=0.01,
#                    headwidth=2.5, headlength=3, headaxislength=2.5)
ax3.streamplot(x3, y3, u_int3_plot, v_int3_plot, density=1.5, 
               color='grey',linewidth=0.5)

# Add heliport
heliports_x_loc = [551]
heliports_y_loc = [505]
ax3.plot(heliports_x_loc, heliports_y_loc, markerfacecolor='black', marker='o', markersize=6,
         markeredgecolor='black', linestyle='none')

# Add a colorbar
v = np.linspace(0, 1.5, 11) # Define range vector
axes = [ax1, ax2, ax3]
bar = plt.colorbar(con,ax=axes,orientation="horizontal",ticks=v, pad=0.1, aspect=25)
bar.set_label("TKE [m$^2$ s$^{-2}]$", fontsize=10)
#bar.ax.tick_params(labelsize=12) # Set tick label size of the colorbar



### Save the figure ###
fig.savefig("/localdata/giersch/AG_Raasch/DroneWaves/Scripts/"
            "xy_TKE_terrain_following.png",dpi=my_dpi,orientation='portrait',format='png', bbox_inches='tight')

