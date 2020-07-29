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
from functools import partial
CONST = K()

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
        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle3())
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
        geom = [


                [15,30],
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