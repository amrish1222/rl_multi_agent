# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:14:44 2020

@author: amris
"""
import skgeom as sg
import numpy as np
from constants import CONSTANTS as K
from matplotlib.path import Path
from collections import defaultdict
import copy
from functools import partial
CONST = K()
from shapely.geometry import Polygon
import math

import time

from visibility import Visibility

class Obstacle:
    def __init__(self):
        pass
    
    def getAllObs_vsbs(self, emptyMap):
        obsMaps = []
        vsbs = []
        vsbPolys = []
        numOpenCellsArr = []
        a = time.time()
        
#        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle1())
#        obsMaps.append(mp)
#        vsbs.append(vsb)
#        vsbPoly =  self.getVisibilityPolys(vsb, mp)
#        vsbPolys.append(vsbPoly)
#        numOpenCellsArr.append(np.count_nonzero(mp==0))
#        
        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle2())
        obsMaps.append(mp)
        vsbs.append(vsb)
        vsbPoly =  self.getVisibilityPolys(vsb, mp)
        vsbPolys.append(vsbPoly)
        numOpenCellsArr.append(np.count_nonzero(mp==0))
#        
#        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle3())
#        obsMaps.append(mp)
#        vsbs.append(vsb)
#        vsbPoly =  self.getVisibilityPolys(vsb, mp)
#        vsbPolys.append(vsbPoly)
#        numOpenCellsArr.append(np.count_nonzero(mp==0))
#        
#        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle4())
#        obsMaps.append(mp)
#        vsbs.append(vsb)
#        vsbPoly =  self.getVisibilityPolys(vsb, mp)
#        vsbPolys.append(vsbPoly)
#        numOpenCellsArr.append(np.count_nonzero(mp==0))
        
#        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle_4R())
#        obsMaps.append(mp)
#        vsbs.append(vsb)
#        vsbPoly =  self.getVisibilityPolys(vsb, mp)
#        vsbPolys.append(vsbPoly)
#        numOpenCellsArr.append(np.count_nonzero(mp==0))

#        mp, vsb = self.getObstacleMap(emptyMap, self.obstacleMR())
#        obsMaps.append(mp)
#        vsbs.append(vsb)
#        vsbPoly =  self.getVisibilityPolys(vsb, mp)
#        vsbPolys.append(vsbPoly)
#        numOpenCellsArr.append(np.count_nonzero(mp==0))
 
#        mp, vsb = self.getObstacleMap(emptyMap, self.rand_obs())
#        obsMaps.append(mp)
#        vsbs.append(vsb)
#        vsbPoly =  self.getVisibilityPolys(vsb, mp)
#        vsbPolys.append(vsbPoly)
#        numOpenCellsArr.append(np.count_nonzero(mp==0))
        
        b = time.time()
        print("create vsb Polys:", round(1000*(b-a), 3))
        return obsMaps, vsbs, vsbPolys, numOpenCellsArr
    
    def getObstacleMap(self, emptyMap, obstacleSet):
        obsList = obstacleSet
        vsb  = Visibility(emptyMap.shape[0], emptyMap.shape[1])
        for obs, isHole in obsList:
            vsb.addGeom2Arrangement(obs)
        
        isHoles = [obs[1] for obs in obsList]
        if any(isHoles) == True:
            pass
        else:
            vsb.boundary2Arrangement(vsb.length, vsb.height)
        
        # get obstacle polygon
        points = CONST.GRID_CENTER_PTS
        img = np.zeros_like(emptyMap, dtype = bool)
        for obs, isHole in obsList:
            p = Path(obs)
            grid = p.contains_points(points)
            mask = grid.reshape(CONST.MAP_SIZE,CONST.MAP_SIZE)
            img = np.logical_or(img , (mask if not isHole else np.logical_not(mask)))
           
        img = img.T
        img = np.where(img,150,emptyMap)
        return img, vsb
    
    def getVisibilityPolys(self,vsb, obsMap):
        polys = defaultdict(partial(np.ndarray, 0))
        for pt in CONST.GRID_CENTER_PTS:
            if not obsMap[int(pt[0]),int(pt[1])] == 150:
                polys[(pt[0],pt[1])] = vsb.getVsbPoly(pt)
            
        return polys
    
    def getVisibilitySets(self, emptyMap):
        obsMaps, vsbs, vsbPolys, numOpenCellsArr = self.getAllObs_vsbs(emptyMap)
        obsMap = obsMaps[0]
        vsb = vsbs[0]
        polys = defaultdict(partial(np.ndarray, 0))
        for pt in CONST.GRID_CENTER_PTS:
            if not obsMap[int(pt[0]),int(pt[1])] == 150:
                polys[(pt[0],pt[1])] = vsb.getVsbPoly(pt)
        
        free_map = np.argwhere(obsMap != 150)
        
        visibility_sets = defaultdict(list)
        for pt in polys:
            points = CONST.GRID_CENTER_PTS
            img = np.zeros_like(emptyMap, dtype = bool)
            p = Path(polys[pt])
            grid = p.contains_points(points)
            mask = grid.reshape(CONST.MAP_SIZE,CONST.MAP_SIZE)
            
            g = (int(pt[0]), int(pt[1]))
            r = int((CONST.LOCAL_SZ -1) /2)
            lx = int(max(0, g[1] - r))
            hx = int(min(CONST.MAP_SIZE, r + g[1] + 1))
            ly = int(max(0, g[0] - r))
            hy = int(min(CONST.MAP_SIZE, r + g[0] + 1))
            tempMask = np.zeros_like(mask).astype(bool)
            tempMask[lx: hx , ly : hy] = True
            
            img = np.logical_and(mask , tempMask)
            
            img = img.T
            
            visibility_set_at_pt = np.argwhere(img)
            temp_set = visibility_set_at_pt.T
            visibility_sets[g] = list(zip(temp_set[0], temp_set[1]))
        
        return len(free_map), visibility_sets
    

    
    def getObstacles(self):
        obstacle = self.obstacle1()
        return obstacle
    
    def obstacle1(self):
        obsList = []
        # add points in CW order and 
        isHole = False
        geom = [[7,0],
                [7,20],
                [29,20],
                [29,29],
                [10,29],
                [10,40],
                [30,40],
                [30,39],
                [11,39],
                [11,30],
                [30,30],
                [30,19],
                [8,19],
                [8,0]]
        obsList.append([geom, isHole])
        
        geom = [[10,50],
                [10,45],
                [25,45],
                [25,46],
                [11,46],
                [11,50]]
        obsList.append([geom, isHole])

        
        geom = [[5,19],
                [5,35],
                [6,35],
                [6,20],
                [7,20],
                [7,19]]
        obsList.append([geom, isHole])
        
        return obsList
        
    
    def obstacle2(self):
        obsList = []
        # add points in CW order and 
        isHole = False
        geom = [[6,6],
                [6,12],
                [44,12],
                [44,6]]
        obsList.append([geom, isHole])
        
        geom = [[35,18],
                [35,23],
                [41,23],
                [41,18]]
        obsList.append([geom, isHole])
        
        geom = [[39,29],
                [39,34],
                [44,34],
                [44,29]]
        obsList.append([geom, isHole])
        
        geom = [[12,29],
                [12,39],
                [17,39],
                [17,29]]
        obsList.append([geom, isHole])
        
        geom = [[23,39],
                [23,44],
                [28,44],
                [28,39]]
        obsList.append([geom, isHole])
        
        return obsList
    
    def obstacle3(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [[0, 5],
                [0, 15],
                [3, 15],
                [3, 20],
                [0, 20],
                [0, 30],
                [3, 30],
                [3, 35],
                [0, 35],
                [0, 45],
                [9, 45],
                [6, 45],
                [6, 35],
                [6, 30],
                [15, 30],
                [15, 35],
                [12, 35],
                [12, 45],
                [21, 45],
                [21, 35],
                [18, 35],
                [18, 30],
                [27, 30],
                [27, 35],
                [24, 35],
                [24, 45],
                [33, 45],
                [33, 35],
                [30, 35],
                [30, 30],
                [39, 30],
                [39, 35],
                [36, 35],
                [36, 45],
                [50, 45],
                [50, 35],
                [42, 35],
                [42, 30],
                [50, 30],
                [50, 20],
                [42, 20],
                [42, 15],
                [50, 15],
                [50, 5],
                [36, 5],
                [36, 15],
                [39, 15],
                [39, 20],
                [30, 20],
                [30, 15],
                [33, 15],
                [33, 5],
                [24, 5],
                [24, 15],
                [27, 15],
                [27, 20],
                [18, 20],
                [18, 15],
                [21, 15],
                [21, 5],
                [12, 5],
                [12, 15],
                [15, 15],
                [15, 20],
                [6, 20],
                [6, 15],
                [9, 15],
                [9, 5]]
        obsList.append([geom, isHole])

        return obsList

    def obstacle_4R(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [[15, 30],
                [15, 35],
                [12, 35],
                [12, 45],
                [21, 45],
                [21, 35],
                [18, 35],
                [18, 30],
                [27, 30],
                [27, 35],
                [24, 35],
                [24, 45],
                [33, 45],
                [33, 35],
                [30, 35],
                [30, 30],
                [30, 20],
                [30, 15],
                [33, 15],
                [33, 5],
                [24, 5],
                [24, 15],
                [27, 15],
                [27, 20],
                [18, 20],
                [18, 15],
                [21, 15],
                [21, 5],
                [12, 5],
                [12, 15],
                [15, 15],
                [15, 20]
                ]
        obsList.append([geom, isHole])

        return obsList
    
    def obstacle4(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[3,30],
                [18,30],
                [18,12],
                [33,12],
                [33,27],
                [30,27],
                [30,30],
                [33,30],
                [33,39],
                [36,39],
                [36,33],
                [39,33],
                [39,36],
                [48,36],
                [48,27],
                [39,27],
                [39,30],
                [36,30],
                [36,21],
                [39,21],
                [39,24],
                [48,24],
                [48,15],
                [39,15],
                [39,18],
                [36,18],
                [36,12],
                [48,12],
                [48,9],
                [18,9],
                [18,6],
                [14,6],
                [14,27],
                [3,27]]
        obsList.append([geom, isHole])
        
        return obsList
    
    def obstacle5(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[6,6],
                [6,30],
                [44,30],
                [44,6],
                [42,6],
                [42,10],
                [40,10],
                [40,6]]
        obsList.append([geom, isHole])
        
        return obsList
    
    def obstacle2R(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[15,30],
                [18,30],
                [30,30],
                [30, 20],
                [30,15],
                [33,15],
                [33,5],
                [24,5],
                [24,15],
                [27,15],
                [27,20],
                [18,20],
                [18,15],
                [21,15],
                [21,5],
                [12,5],
                [12,15],
                [15,15],
                [15,20],
               ]
        obsList.append([geom, isHole])
        
        return obsList
    
    def twist_mirror(self, geom):
        
        for g in geom:
            g[0], g[1] = 100 - g[0], 100 - g[1]
    
    def obstacleMR(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[5,5],
                [5,26],
                [15,26],
                [15,29],
                [5,29],
                [5,39],
                [25,39],
                [25,34],
                [28,34],
                [28,46],
                [25,46],
                [25,43],
                [5,43],
                [5,57],
                [25,57],
                [25,54],
                [28,54],
                [28,66],
                [25,66],
                [25,61],
                [5,61],
                [5,71],
                [15,71],
                [15,74],
                [5,74],
                [5,95],
                [20,95],
                [20,83],
                [25,83],
                [25,73],
                [35,73],
                [35,76],
                [28,76],
                [28,86],
                [23,86],
                [23,95],
                [47,95],
                [47,76],
                [44,76],
                [44,73],
                [56,73],
                [56,76],
                [53,76],
                [53,95],
                [77,95],
                [77,86],
                [72,86],
                [72,76],
                [65,76],
                [65,73],
                [75,73],
                [75,83],
                [80,83],
                [80,95],
                [95,95]]
        
        geom_temp = copy.deepcopy(geom)
        self.twist_mirror(geom_temp)
        geom += geom_temp[1:-1]
                
        obsList.append([geom, isHole])
        
        isHole = False
        geom = [[40,40],
                [40,60],
                [60,60],
                [60,40]]
        obsList.append([geom, isHole])
        
        return obsList
    
    def rand_obs(self):
        map_bounds = CONST.MAP_SIZE-1
        
        polygons = []
        obsList = []
        
        # obs 1
        
        for i in range(4):
            intersecting_others = True
            fail_counter = 0
            while(intersecting_others and fail_counter <= 50):
                fail_counter += 1
                geom, isHole, center, bbox = self.base_obs1(np.random.randint(5,16),0)
                spawn_bounds = np.array([[0,0],[map_bounds, map_bounds]]) + np.array([[center[0], center[1]],[-center[0], -center[1]]])
                
                offset_x = center[0]%1
                offset_y = center[1]%1
                
                rand_x = offset_x + int(np.random.uniform(spawn_bounds[:,0][0], spawn_bounds[:,0][1]))
                rand_y = offset_y + int(np.random.uniform(spawn_bounds[:,1][0], spawn_bounds[:,1][1]))
                
#                print("obs1",rand_x, rand_y)
                
                geom += np.array([rand_x- center[0], rand_y - center[1]])
                
                cur_polygon = Polygon(geom)
                
                if len(polygons) == 0:
                    polygons.append(cur_polygon)
                    obsList.append([np.ndarray.tolist(geom), isHole])
                    intersecting_others = False
                else:
                    intersecting_others = False
                    for polygon in polygons:   
                        if polygon.intersects(cur_polygon):
                            intersecting_others = True
                            break
                    if not intersecting_others:
                        intersecting_others = False
                        polygons.append(cur_polygon)
                        obsList.append([np.ndarray.tolist(geom), isHole])
                        
        for i in range(2):
            intersecting_others = True
            fail_counter = 0
            while(intersecting_others and fail_counter <= 50):
                fail_counter += 1
                geom, isHole, center, bbox = self.base_obs2(np.random.randint(5,16),np.random.randint(0,5))
                spawn_bounds = np.array([[0,0],[map_bounds, map_bounds]]) + np.array([[center[0], center[1]],[-center[0], -center[1]]])
                
                offset_x = center[0]%1
                offset_y = center[1]%1
                
                rand_x = offset_x + int(np.random.uniform(spawn_bounds[:,0][0], spawn_bounds[:,0][1]))
                rand_y = offset_y + int(np.random.uniform(spawn_bounds[:,1][0], spawn_bounds[:,1][1]))
                
#                print("obs2",rand_x, rand_y)

                geom += np.array([rand_x- center[0], rand_y - center[1]])
                
                cur_polygon = Polygon(geom)
                
                if len(polygons) == 0:
                    polygons.append(cur_polygon)
                    obsList.append([np.ndarray.tolist(geom), isHole])
                    intersecting_others = False
                else:
                    intersecting_others = False
                    for polygon in polygons:   
                        if polygon.intersects(cur_polygon):
                            intersecting_others = True
                            break
                    if not intersecting_others:
                        intersecting_others = False
                        polygons.append(cur_polygon)
                        obsList.append([np.ndarray.tolist(geom), isHole])
        return obsList
    
    def bbox(self,geom):
        
        geom = np.asarray(geom)
        minx = np.min(geom[:,0])
        miny = np.min(geom[:,1])
        maxx = np.max(geom[:,0])
        maxy = np.max(geom[:,1])
        
        return [[minx, miny],[maxx,maxy]]
    
    def base_obs1(self, base_length, rotation = 0):
        #box
        s = base_length
        isHole = False
        geom = [[0,0],
                [0,s],
                [s,s],
                [s,0]]
        
#        geom = [[s,s],
#                [s,0],
#                [0,0],
#                [0,s]]
        
        bbox = self.bbox(geom)
        
        center = np.mean(np.array(bbox), axis = 0)
        
        return geom, isHole, center, bbox
    
    def base_obs2(self, base_length, rotation = 0):
        #L
        s = base_length
        isHole = False
        geom = [[0,0],
                [0,2*s],
                [s,2*s],
                [s,s],
                [2*s,s],
                [2*s,0]]
        
        bbox = self.bbox(geom)
        
        center = np.mean(np.array(bbox), axis = 0)
        
        if not rotation == 0:
            rot_geom = []
            for pt in geom:
                rot_geom.append(self.rotatePoint(center, pt, rotation * 90))
            geom = rot_geom
        return geom, isHole, center, bbox   
    
    def rotatePoint(self, centerPoint,point,angle):
        """Rotates a point around another centerPoint. Angle is in degrees.
            Rotation is counter-clockwise"""
        angle = math.radians(angle)
        temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
        temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
        temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
        return temp_point
        