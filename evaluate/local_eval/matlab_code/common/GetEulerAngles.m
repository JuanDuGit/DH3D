function [rx, ry, rz]= GetEulerAngles(R)
% This function return the rotation along x,y and z direction from a 
% Rotation Matrix
%Inputs:
    % R= 3x3 Rotation Matrix
%Outputs:
    % rx= Rotation along x direction in radians
    % ry= Rotation along y direction in radians
    % rz= Rotation along z direction in radians
    
%     R =
%  
% [                           cos(ry)*cos(rz),                          -cos(ry)*sin(rz),          sin(ry)]
% [ cos(rx)*sin(rz) + cos(rz)*sin(rx)*sin(ry), cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz), -cos(ry)*sin(rx)]
% [ sin(rx)*sin(rz) - cos(rx)*cos(rz)*sin(ry), cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz),  cos(rx)*cos(ry)]
% Author : Sandeep Sasidharan
% http://sandeepsasidharan.webs.com
ry=asin(R(1,3));
rz=acos(R(1,1)/cos(ry));
rx=acos(R(3,3)/cos(ry));