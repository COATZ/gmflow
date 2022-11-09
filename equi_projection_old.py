# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import matplotlib.pyplot as plt

# equirectangular_img = plt.imread("/home/cartizzu/Documents/2_CODE/1_PHOTOS/castle_equi.jpg")

# plt.imshow(equirectangular_img)

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mpl_sphere(img,grid_center):

    size_img = np.shape(img)
    width = size_img[1]
    height = size_img[0]

    # define a grid matching the map size, subsample along with pixels
    theta = np.linspace(0, np.pi, img.shape[0])
    phi = np.linspace(0, 2*np.pi, img.shape[1])

    count = 180 # keep 180 points along theta and phi
    theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
    phi_inds = np.linspace(0, img.shape[1] - 1, count*2).round().astype(int)
    theta = theta[theta_inds]
    phi = phi[phi_inds]
    img = img[np.ix_(theta_inds, phi_inds)]

    theta,phi = np.meshgrid(theta, phi)
    R = 1

    # sphere
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # create 3d Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1) # we've already pruned ourselves
    ax.grid()
    # make the plot more spherical
    #ax.axis('scaled')
    
    theta_center = np.deg2rad(90-grid_center[1]*180/height)
    phi_center = np.deg2rad(grid_center[0]*360/width-180)
    
    print(np.rad2deg(theta_center),np.rad2deg(phi_center))
    
    R = 1.5
    grid_center_sph = [R * np.sin(theta_center) * np.cos(phi_center),R * np.sin(theta_center) * np.sin(phi_center),R * np.cos(theta_center)]
    
    print(grid_center_sph)
    
    ax.scatter(grid_center_sph[0],grid_center_sph[1],grid_center_sph[2],color='r', marker='o')

    
    #pu = noname(img,5000,2500,x,y,z)
    #ax.scatter(pu[0],pu[1],pu[2],color="r",s=200)
    #ax.view_init(elev=0, azim=0)
    # make the plot more spherical
    #ax.axis('scaled')

def mpl_plane(img):
    plt.figure(2)
    plt.imshow(img)
    plt.scatter(5000,2500,color="r",s=200)


def noname(img,x,y,xr,yr,zr):
    h = img.shape[0]
    w = img.shape[1]
    
    long = (x-w/2)*2*np.pi/w
    lat = (h/2-y)*np.pi/h
    print(long, lat)

    pu = [0,0,0]
    pu[0] = np.cos(long)*np.cos(lat)
    pu[1] = np.sin(long)*np.cos(lat)
    pu[2] = np.sin(lat)
    
    tx = np.cross([0,0,1],pu)
    ty = np.cross(pu,tx)
    print(tx,ty)

    rho = np.tan(2*np.pi/w)
    print(rho)
      
    rloc = [zr,xr,-yr]
    r_sph = rho * (tx*rloc+ty*rloc)
    print(r_sph)
    

    return pu


def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': return np.dot(X, np.array([
    [1.,  0.,  0.],
    [0. ,  c, -s],
    [0. ,  s,  c]
  ]))
  elif axis == 'y': return np.dot(X, np.array([
    [c,  0,  -s],
    [0,  1,   0],
    [s,  0,   c]
  ]))
  elif axis == 'z': return np.dot(X, np.array([
    [c, -s,  0 ],
    [s,  c,  0 ],
    [0,  0,  1.],
  ]))

def compute_xy_equi(center0,r_0,alpha,r_scale):
    omega0 = (center0[0]-width/2)*2*np.pi/width
    phi0 = -(center0[1]-height/2)*np.pi/height
    
    print("###### TATENO ########")
    print(phi0,omega0)
    print(np.rad2deg(phi0),np.rad2deg(omega0))
    
    pu = [np.cos(phi0)*np.sin(omega0),np.sin(phi0),np.cos(phi0)*np.cos(omega0)]
    print("Pu = ",pu)
    tx = np.cross([0,1,0],pu)
    ty = np.cross(pu,tx)
    print("tx = ",tx," ty = ",ty)
    
    r=[]
    for i in np.arange(-(r_0-1)/2,(r_0-1)/2+1,1):
        for j in np.arange(-(r_0-1)/2,(r_0-1)/2+1,1):
            if not (i==0 and j==0):
                r.append([r_scale*i,r_scale*j,1])
    #print(r)
    ref_ij=[]           
    for elt in r:
        ref_ij.append([center0[0]+elt[0],center0[1]+elt[1]])
    
    pur_ij=[]        
    d=r_0/(2*np.tan(alpha/2))
    for elt in r:
        pur_ij.append(pu + np.tan(2*np.pi/width)*(tx*elt[0]+ty*elt[1]))
        
    print("toto = ",np.tan(2*np.pi/width))
    print(pur_ij)
    print("##########################################")
    omega_r_ij=[]
    phi_r_ij=[]
    for elt in pur_ij:
        if elt[0]>=0:
            omega_r_ij.append(np.arctan2(np.array(elt[2]),np.array(elt[0]))-np.pi/2)
        else:
            omega_r_ij.append(np.arctan2(np.array(elt[2]),np.array(elt[0]))-np.pi/2)
        phi_r_ij.append(np.arcsin(elt[1]))
    
    print(omega_r_ij)
    print(phi_r_ij)

    plt.figure(6)
    plt.scatter(omega_r_ij,phi_r_ij)
    plt.scatter(omega0,phi0)
    plt.show()    
    
    xr_ij = []
    for elt in omega_r_ij:
        xr_ij.append((elt/(2*np.pi)+1/2)*width)
    yr_ij = []
    for elt in phi_r_ij:
        yr_ij.append((1/2-elt/np.pi)*height)
         
    plt.figure(5)
    plt.gca().invert_yaxis()
    plt.scatter(xr_ij,yr_ij)
    plt.show()    
    
    print(xr_ij)
    print(yr_ij)
    return xr_ij,yr_ij,pur_ij,ref_ij

from scipy.spatial import ConvexHull
def convexhull(p):
    p = np.array(p)
    hull = ConvexHull(p)
    return p[hull.vertices,:]

def ccw_sort(p):
    p = np.array(p)
    mean = np.mean(p,axis=0)
    d = p-mean
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def show_clara_proj(center0,r,alpha,r_scale):
    d=1/(2*np.tan(alpha/2))
    print("###### CLARA ########")
    print("d =",d)
    print("d =",1/d)
    
    phi0 = (center0[0]-width/2)*2*np.pi/width
    omega0 = -(center0[1]-height/2)*np.pi/height
    print("PHI0, OMEGA0 = ",phi0,omega0)
    print("in deg = ",np.rad2deg(phi0),np.rad2deg(omega0))
    
    R_phi0 = R.from_euler('y', phi0).as_matrix()
    R_omega0 = R.from_euler('x', -omega0).as_matrix()
    #print(R_phi0)
    #print(R_omega0)
    
    ptild= []
    p = []
    for j in np.arange((r-1)/2,-(r-1)/2-1,-1):
        for i in np.arange(-(r-1)/2,(r-1)/2+1,1):
                ptild.append([r_scale*i,r_scale*j,d])
                p.append(np.dot(R_phi0,np.dot(R_omega0,[r_scale*i,r_scale*j,d]))/np.linalg.norm([r_scale*i,r_scale*j,d]))
                temp2=[np.cos(phi0)*r_scale*i+np.sin(phi0)*(np.sin(-omega0)*r_scale*j+np.cos(-omega0)*d)
                       ,np.cos(-omega0)*r_scale*j-np.sin(-omega0)*d,
                       -np.sin(phi0)*r_scale*i+np.cos(phi0)*(np.sin(-omega0)*r_scale*j+np.cos(-omega0)*d)]
                #print(temp2)
                #print(np.dot(R_phi0,np.dot(R_omega0,[r_scale*i,r_scale*j,d]))/np.linalg.norm([r_scale*i,r_scale*j,d]))
                #print("temp2 = ",temp2/np.sqrt((r_scale*i)*(r_scale*i)+(r_scale*j)*(r_scale*j)+d*d))
    print("ptild = ",ptild)
    print("pm  = ",np.dot(R_phi0,np.dot(R_omega0,[0,0,d]))/np.linalg.norm([r_scale*0,r_scale*0,d]))
    print("pm2 = ",np.dot(R_phi0,np.dot(R_omega0,[1,0,d]))/np.linalg.norm([r_scale*1,r_scale*0,d]))
    print(p)
    
    phi_ij = []
    omega_ij = []
    u_ij = []
    v_ij = []   
    for elt in p:
        temp0 = np.arctan2(np.array(elt[0]),np.array(elt[2]))
        temp1 = np.arcsin(np.array(elt[1]))
        phi_ij.append(temp0)
        omega_ij.append(temp1)
        u_ij.append((temp0/(2*np.pi)+1/2)*width)
        v_ij.append((-temp1/np.pi+1/2)*height)
        print("X_temp, Z_temp :",elt[0],elt[2])
        print("phi_temp, omega_temp :",temp0,temp1)
        print("new_r, new_c :",(-temp1/np.pi+1/2)*height,(temp0/(2*np.pi)+1/2)*width)
     
    print(phi_ij)
    print(omega_ij)
    print("u_ij, v_ij = ",u_ij,v_ij)
    
    plt.figure(5)
    plt.scatter(np.rad2deg(phi_ij),np.rad2deg(omega_ij))
    plt.show()
    
    s_size = 5
    ref_ij=[]           
    for elt in ptild:
        ref_ij.append([center0[0]+elt[0],center0[1]+elt[1]])    
        
    
    plt.imshow(img)    
    for elt in ref_ij:
        plt.scatter(elt[0],elt[1],color='g',s=s_size)
    #plt.scatter(center0[0],center0[1],color='b',s=s_size)
    plt.scatter(u_ij,v_ij,color='r',s=s_size)
    p_poly_r0=[]
    u_loc = []
    for idx,elt in enumerate(u_ij):
        p_poly_r0.append([u_ij[idx],v_ij[idx]]) 
        print("uij  ="+str(u_ij[idx])+"   vij    =   "+str(v_ij[idx]))
        u_loc.append([u_ij[idx],v_ij[idx]])        
    #poly_r0 = plt.Polygon(ccw_sort(p_poly_r0), ec="r",facecolor="none")
    #ax.add_patch(poly_r0)

    plt.figure(4)
    plt.gca().invert_yaxis()
    plt.scatter(u_ij,v_ij)
    #plt.scatter(center0[0],center0[1])
    plt.show()  
        
    for j in np.arange(0,r,1):
        for i in np.arange(0,r,1):
            print("i j :", j,i)
            print("NEW PHI THETA :",omega_ij[i+j*r],phi_ij[i+j*r])
            print("NEW R C :",v_ij[i+j*r],u_ij[i+j*r])

    return p,u_loc

def deformable_im2col_bilinear(bottom_data, data_width, data_height, h, w, index):

    h_low = int(np.floor(h));
    w_low = int(np.floor(w));

    h_high = h_low + 1;
    w_high = w_low + 1;

    if(h_low<0):
        h_low = h_low + data_height
    if(w_low<0):
        w_low = w_low + data_width
    if(h_high>=data_height):
        h_high= h_high - data_height
    if(w_high>=data_width):
        w_high = w_high - data_width

    u = h - h_low;
    v = w - w_low;
    v1 = bottom_data[h_low,w_low];
    v2 = bottom_data[h_low,w_high];
    v3 = bottom_data[h_high,w_low];
    v4 = bottom_data[h_high,w_high];

    val = (1-u)*(1-v)*v1 + (1-u)*v*v2 + u*(1-v)*v3 + u*v*v4;
        
    return val   

from scipy.spatial.transform import Rotation as R
if __name__ == "__main__":
    image_file = '/home/cartizzu/Documents/COPY_FROM_MSI/1_PHD/1_PHOTOS/file0001.png'
    img = plt.imread(image_file)
    size_img = np.shape(img)
    width = size_img[1]
    height = size_img[0]
    #width = 768
    #height = 384
    
    kw = kh = 7
    r = rw = rh = 7
    alpha = alpha_w = alpha_h = 2*np.pi/width
    #print("alpha = ",np.rad2deg(alpha))
    
    center0 = [width/2,height/2]
    center1 = [width/2,height/20]
    center2 = [width/2,height/5]

    # r_scale = 40
      
    # xr_ij1,yr_ij1,pur_ij1,ref_ij1 = compute_xy_equi(center1,r_scale)
    # xr_ij2,yr_ij2,pur_ij2,ref_ij2 = compute_xy_equi(center2,r_scale)
    
    # s_plot=5
    # plt.figure(1)
    # ax = plt.subplot(111)
    # plt.imshow(img)
    # plt.scatter(center0[0],center0[1],color='b',s=s_plot)
    # # p_poly_g0 = []
    # # for elt in ref_ij0:
    # #     #if not (elt[0] < 0 or elt[0] > width or elt[1] < 0 or elt[1] > height):
    # #     plt.scatter(max(elt[0],0),max(0,elt[1]),color='g',s=s_plot)
    # #     p_poly_g0.append([max(elt[0],0),max(0,elt[1])])
    # # poly_g0 = plt.Polygon(convexhull(p_poly_g0), ec="g",facecolor="none")
    # # ax.add_patch(poly_g0)
    # plt.scatter(xr_ij0,yr_ij0,color='r',s=s_plot)
    # p_poly_r0 = []
    # for idx,elt in enumerate(xr_ij0):
    #     p_poly_r0.append([xr_ij0[idx],yr_ij0[idx]])
    # poly_r0 = plt.Polygon(convexhull(p_poly_r0), ec="r",facecolor="none")
    # ax.add_patch(poly_r0)
    
    # plt.scatter(center1[0],center1[1],color='b',s=s_plot)
    # plt.scatter(xr_ij1,yr_ij1,color='r',s=s_plot)
    # p_poly_r1 = []
    # for idx,elt in enumerate(xr_ij1):
    #     p_poly_r1.append([xr_ij1[idx],yr_ij1[idx]])
    # poly_r1 = plt.Polygon(convexhull(p_poly_r1), ec="r",facecolor="none")
    # ax.add_patch(poly_r1)

    # plt.scatter(center2[0],center2[1],color='b',s=s_plot)
    # plt.scatter(xr_ij2,yr_ij2,color='r',s=s_plot)
    # p_poly_r2 = []
    # for idx,elt in enumerate(xr_ij2):
    #     p_poly_r2.append([xr_ij2[idx],yr_ij2[idx]])
    # poly_r2 = plt.Polygon(convexhull(p_poly_r2), ec="r",facecolor="none")
    # ax.add_patch(poly_r2)
    
    # plt.axis('off')
    
    # #plt.scatter(center1[0],center1[1],color='b',s=s_plot)
    # #plt.scatter(xr_ij1,yr_ij1,color='r',s=s_plot)
    # plt.savefig('C:/Users/charl_000/Documents/2_CODE/1_PHOTOS/out.jpg',dpi=1200, bbox_inches='tight', pad_inches=0)
    
    # plt.figure(2)
    # fig = plt.figure(figsize = (10,10))
    # ax = fig.add_subplot(111, projection='3d')   
    # u = np.linspace(0, 2 * np.pi, 200)
    # v = np.linspace(0, np.pi, 200)    
    # x = 1 * np.outer(np.cos(u), np.sin(v))
    # y = 1 * np.outer(np.sin(u), np.sin(v))
    # z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    # ax.plot_surface(x, y, z, facecolors=img/255, cstride=1, rstride=1)  
    # #ax.plot_wireframe(x, y, z, rstride=10, cstride=10,linewidth=1) 
    # for elt in pur_ij0:
    #     ax.scatter(elt[0],elt[1],elt[2], color='r')
    # #ax.scatter(pur_ij1[0],pur_ij1[1],pur_ij1[2], color='b')
    # plt.show()
    

    
    center0 = [3*width/4,height/12]
    center1 = [230,height/2]
    center2 = [2*width/5,2*height/8]
    center0 = [40,20]
    center0 = [768,350]
    #center0 = [0,0]
    
    r_scale = 1
    #xr_ij0,yr_ij0,pur_ij0,ref_ij0 = compute_xy_equi(center0,r,alpha,r_scale) 
    
    # plt.figure(1)
    # ax = plt.subplot(111)
    
    p0,u_loc = show_clara_proj(center0,r,alpha,r_scale)
    # # p1 = show_clara_proj(center1,r,alpha)
    # # p2 = show_clara_proj(center2,r,alpha)

    toto = u_loc
    print(toto)
    for elt in u_loc:
        val = deformable_im2col_bilinear(img, width, height, elt[1], elt[0], 1)
        print(elt[1], elt[0])
        print(val)

    
    # plt.axis('off')
    # plt.savefig('/home/cartizzu/Documents/1_PHD/1_PHOTOS/out3.jpg',dpi=1200, bbox_inches='tight', pad_inches=0)
    
    
    # # define a grid matching the map size, subsample along with pixels
    # phi = np.linspace(-np.pi/2, np.pi/2, height)
    # theta = np.linspace(-np.pi, np.pi, width)
    # count = 250 # keep 180 points along theta and phi
    # theta_inds = np.linspace(0, width - 1, count).round().astype(int)
    # phi_inds = np.linspace(0, height - 1, count).round().astype(int)
    # theta = theta[theta_inds]
    # phi = phi[phi_inds]
    # img = img[np.ix_(phi_inds, theta_inds)]
    # theta,phi = np.meshgrid(theta, phi)
    # R = 0.95
    # # sphere
    # x = R * np.sin(theta) * np.cos(phi)
    # y = R * np.sin(phi)
    # z = R * np.cos(theta) * np.cos(phi)
    # # create 3d Axes
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(z.T, y.T, -x.T, facecolors=img, cstride=1, rstride=1) # we've already pruned ourselves
    # #ax.plot_wireframe(z.T, x.T, y.T, rstride=2, cstride=2, linewidth=1)
    # for elt in p0:
    #       ax.scatter(elt[2],elt[0],elt[1], color='r')          
    # for elt in p1:
    #       ax.scatter(elt[2],elt[0],elt[1], color='r')
    # for elt in p2:
    #       ax.scatter(elt[2],elt[0],elt[1], color='r')
    # ax.grid()
    # ax.view_init(elev=0, azim=0)
    # plt.axis('off')
    # plt.savefig('/home/cartizzu/Documents/1_PHD/1_PHOTOS/sphere_out.jpg',dpi=1200, bbox_inches='tight', pad_inches=0)
  