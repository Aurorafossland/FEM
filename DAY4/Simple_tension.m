% Inputfile for Assignment1 ex2
% 2025 Prepared by Casper Schousboe Andreasen
% mprop contains also Signorini material model parameters

% Node coordinates: x, y
X = [
0	0
1.5	0
3	0
];
% Element connectivity: node1_id, node2_id, material_id
IX = [
1	2	1
2	3	1
];
% Element properties: Young's modulus, area
mprop = [
1e6	1 ];
% Nodal diplacements: node_id, degree of freedom (1 - x, 2 - y), displacement
bound = [
1	2	0
1	1	0
2	2	0
3   2   0
];
% Nodal loads: node_id, degree of freedom (1 - x, 2 - y), load
loads = [
3	1	200
];
% Control parameters
plotdof = 24;

n_incr = 10;
