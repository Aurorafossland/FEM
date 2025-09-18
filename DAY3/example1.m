% File example1.m, DAY1, modified 3/9 by OS
%
% Example: 3-bar truss
% No. of Nodes: 3  
% No. of Elements : 3
clear all

% Coordinates of 3 nodes,
X = [  0.00  0.00 
       2.00  0.00 
       4.00  0.00 
       6.00  0.00 
       8.00  0.00
       10.00  0.00 
       9.00  1.00 
       7.00  1.00 
       5.00  1.00 
       3.00  1.00
       1.00  1.00 
       ];

% Topology matrix IX(node1,node2,propno),
IX = [ 1  2  1 
       2  3  1
       3  4  1 
       4  5  1
       5  6  1
       6  7  2 
       7  8  1
       8  9  1
       9  10  1
       10  11  1 
       11  1  2 
       11  2  2 
       2  10  2
       10  3  2
       3  9  2 
       9  4  2
       4  8  2
       8  5  2
       5  7   2 ]; 
      
% Element property matrix mprop = [ E A ],
mprop = [ 210E+9 0.0002
          70E+9  0.0001]; 

% Prescribed loads mat(node,ldof,force)
loads = [ 4   2 -5000
          5   2 -10000];

% Boundary conditions mat(node,ldof,disp)   
bound = [ 1  2  0.0
          3  2  0.0
          3  1  0.0 ];

% Control Parameters
plotdof = 19;
