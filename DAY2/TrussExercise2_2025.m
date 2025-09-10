% Inputfile for Assignment1 ex2
% 2025 Prepared by Casper Schousboe Andreasen
% mprop contains also Signorini material model parameters

% Node coordinates: x, y
X = [
0	0
0	1
0	2
0	3
1	1
1	2
1	3
2	2
2	3
3	2
3	3
4	2
4	3
5	2
5	3
6	1
6	2
6	3
7	0
7	1
7	2
7	3
];
% Element connectivity: node1_id, node2_id, material_id
IX = [
2	1	1
5	1	1
3	2	1
6	2	1
5	2	1
4	3	1
7	3	1
6	3	1
5	3	1
7	4	1
6	4	1
6	5	1
8	5	1
7	6	1
9	6	1
8	6	1
9	7	1
8	7	1
9	8	1
11	8	1
10	8	1
11	9	1
10	9	1
11	10	1
13	10	1
12	10	1
13	11	1
12	11	1
13	12	1
15	12	1
14	12	1
15	13	1
14	13	1
15	14	1
18	14	1
17	14	1
16	14	1
18	15	1
17	15	1
17	16	1
21	16	1
20	16	1
19	16	1
18	17	1
22	17	1
21	17	1
20	17	1
22	18	1
21	18	1
20	19	1
21	20	1
22	21	1
];
% Element properties: Young's modulus, area, c1, c2, c3, c4
mprop = [
1	1  0.1 70 0.3 100
];
% Nodal diplacements: node_id, degree of freedom (1 - x, 2 - y), displacement
bound = [
1	2	0
19	1	0
19	2	0
];
% Nodal loads: node_id, degree of freedom (1 - x, 2 - y), load
loads = [
12	2	-200
];
% Control parameters
plotdof = 24;
