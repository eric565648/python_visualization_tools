#!/usr/bin/env python

import numpy as np
import copy
from math import atan2, cos, sin, asin, acos, sqrt, pi

class points_registration(object):
    def __init__(self, X, Y):
        super(points_registration, self).__init__()

        self.X = X
        self.Y = Y
        self.num_p = len(self.X)
        self.R = None
        self.theta = None
        self.T = None
        self.error = None
    
    def register(self):
        
        error_arr = np.array([])
        theta_arr = np.array([])
        T_arr = np.zeros((self.num_p, 2))
        for i in range(self.num_p):
            # calculate vectors
            vx = self.calculate_vector(self.X, i)
            vy = self.calculate_vector(self.Y, i)

            # get angle differences
            # sin(A-B) = sinAcosB - cosAsinB
            # cos(A-B) = cosAcosB + sinAsinB
            sin_sum = 0
            cos_sum = 0
            for j in range(self.num_p-1):
                d_theta = self.get_angle_diff(vx[j], vy[j])
                sin_sum += sin(d_theta)
                cos_sum += cos(d_theta)

            # average over differences
            theta = atan2(sin_sum/(self.num_p-1), cos_sum/(self.num_p-1))
            # add theta
            theta_arr = np.append(theta_arr, theta)
            R = np.array([[cos(theta), -sin(theta)],[sin(theta),cos(theta)]])
            R_points = np.dot(R, self.Y.T).T
            T = self.X[i] - R_points[i]
            # add transform
            T_arr[i] = self.X[i] - R_points[i]
            T_points = R_points + T
            # add error
            error_arr = np.append(error_arr, self.calculate_RMSE(self.X, T_points))
        
        min_i = np.argmin(error_arr)
        self.error = error_arr[min_i]
        self.theta = theta_arr[min_i]
        self.R = np.array([[cos(self.theta), -sin(self.theta)],[sin(self.theta),cos(self.theta)]])
        self.T = T_arr[min_i]
        final = np.dot(self.R, self.Y.T).T + self.T

        return final, self.R, self.T, self.theta, self.error
    
    def get_registration(self):

        return self.R, self.T, self.theta, self.error

    def calculate_RMSE(self, X, Y):

        delta = X - Y
        dis_sq = np.array([])
        for i in range(len(delta)):
            dis_sq = np.append(dis_sq ,delta[i][0]**2+delta[i][1]**2)
        return np.sqrt(np.mean(dis_sq))

    def get_angle_diff(self, A, B):

        a_theta = atan2(A[1], A[0])
        b_theta = atan2(B[1], B[0])
        diff = a_theta-b_theta

        while diff < -1*pi:
            diff += 2*pi
        while diff >= pi:
            diff -= 2*pi

        return a_theta - b_theta

    def calculate_vector(self, points, start_p):

        vectors = np.zeros((len(points)-1,2))
        for i in range(len(points)-1):
            this_i = start_p + 1 + i
            if this_i >= len(points):
                this_i -= len(points)
            
            vectors[i] = np.array(points[this_i]-points[start_p])
        
        return vectors

    
    