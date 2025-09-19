% File Exercise_3_2.m, DAY3
%
clear all

% Coordinates of 3 nodes,
X = [  0.00  0.40 
       1.50  0.00 
       3.00  0.40 ];

% Topology matrix IX(node1,node2,propno),
IX = [ 1  2  1 
       2  3  1 ];
      
% Element property matrix mprop = [ E A ],
mprop = [ 1e6 1.0 ];

% Prescribed loads mat(node,ldof,force)
loads = [ 2   2 10000 ];

% Boundary conditions mat(node,ldof,disp)   
bound = [ 1  1  0.0
          1  2  0.0
          3  1  0.0 
          3  2  0.0 ];

% Control Parameters
plotdof = 6;

n_incr = 40;  % Number of increments